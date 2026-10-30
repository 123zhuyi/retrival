/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_copy_simt_dim5.h
 * \brief
 */

#ifndef VIEW_COPY_SIMT_DIM5_H_
#define VIEW_COPY_SIMT_DIM5_H_

#include "view_copy_base.h"

namespace ViewCopy {
using namespace AscendC;

template <typename T, typename U>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void CopyUbToGmDim5(__gm__ T* dst, __ubuf__ T* src, uint32_t batchSize,
    uint32_t ubDim4Stride, uint32_t ubDim3Stride, uint32_t ubDim2Stride, uint32_t ubDim1Stride,
    __ubuf__ uint32_t* ParamDim4MagicUb, __ubuf__ uint32_t* ParamDim4ShiftUb,
    __ubuf__ uint32_t* ParamDim5SrcStrideUb, __ubuf__ U* ParamDim5DstStrideUb)
{
    for (uint32_t i = threadIdx.x; i < batchSize; i += blockDim.x) {
        uint32_t idx = i;
        uint32_t srcIndex = 0;
        U dstIndex = 0;

        uint32_t t4 = __umulhi(idx, ParamDim4MagicUb[DIM0]);
        t4 = t4 + idx;
        uint32_t dim4Index = t4 >> ParamDim4ShiftUb[DIM0];
        uint32_t newIdx4 = idx - dim4Index * ubDim4Stride;
        srcIndex += (dim4Index * ParamDim5SrcStrideUb[DIM4]);
        dstIndex += (dim4Index * ParamDim5DstStrideUb[DIM4]);

        uint32_t t3 = __umulhi(newIdx4, ParamDim4MagicUb[DIM1]);
        t3 = t3 + newIdx4;
        uint32_t dim3Index = t3 >> ParamDim4ShiftUb[DIM1];
        uint32_t newIdx3 = newIdx4 - dim3Index * ubDim3Stride;
        srcIndex += (dim3Index * ParamDim5SrcStrideUb[DIM3]);
        dstIndex += (dim3Index * ParamDim5DstStrideUb[DIM3]);

        uint32_t t2 = __umulhi(newIdx3, ParamDim4MagicUb[DIM2]);
        t2 = t2 + newIdx3;
        uint32_t dim2Index = t2 >> ParamDim4ShiftUb[DIM2];
        uint32_t newIdx2 = newIdx3 - dim2Index * ubDim2Stride;
        srcIndex += (dim2Index * ParamDim5SrcStrideUb[DIM2]);
        dstIndex += (dim2Index * ParamDim5DstStrideUb[DIM2]);

        uint32_t t1 = __umulhi(newIdx2, ParamDim4MagicUb[DIM3]);
        t1 = t1 + newIdx2;
        uint32_t dim1Index = t1 >> ParamDim4ShiftUb[DIM3];
        uint32_t dim0Index = newIdx2 - dim1Index * ubDim1Stride;
        srcIndex += (dim1Index * ParamDim5SrcStrideUb[DIM1]);
        dstIndex += (dim1Index * ParamDim5DstStrideUb[DIM1]);
        
        srcIndex += (dim0Index * ParamDim5SrcStrideUb[DIM0]);
        dstIndex += (dim0Index * ParamDim5DstStrideUb[DIM0]);
        dst[dstIndex] = src[srcIndex];
    }
}

template <typename T>
class ViewCopySimtDim5 : public ViewCopyBase<T> {
public:
    __aicore__ inline ViewCopySimtDim5(TPipe &pipe, const ViewCopyTilingData *tilingData) :
        pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(
        GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
        GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
        GM_ADDR out);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM5> &dmaParam,
        int32_t batchSize);
    __aicore__ inline void CopyIn(const GlobalTensor<T> &src, const MultiCopyParams<T, DIM5> &dmaParams);
private:
    TPipe &pipe_;
    const ViewCopyTilingData *tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;

    GlobalTensor<T> srcGm_;
    GlobalTensor<T> dstGm_;
    TBuf<TPosition::VECCALC> Dim4paramMagicBuf_;
    TBuf<TPosition::VECCALC> Dim4paramShiftBuf_;
    TBuf<TPosition::VECCALC> Dim5paramSrcStrideBuf_;
    TBuf<TPosition::VECCALC> Dim5paramDstStrideBuf_;

    MultiCopyParams<T, DIM5> tailDmaParam_;
    MultiCopyParams<T, DIM5> dmaParam_;
    DataCopyExtParams tailCopyParams_;
    DataCopyExtParams copyParams_;

    int64_t blockOffset_ = 0;
    int32_t batchSize_ = 1;
    int32_t tailBatchSize_ = 1;
    uint32_t dim4UbdstStride_ = 1;
    uint32_t dim3UbdstStride_ = 1;
    uint32_t dim2UbdstStride_ = 1;
    uint32_t dim1UbdstStride_ = 1;
    uint32_t shift_[4];
    uint32_t m_[4];
};

template <typename T>
__aicore__ inline void ViewCopySimtDim5<T>::Init(
    GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
    GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
    GM_ADDR out)
{
    this->ParseTilingData(tilingData_);
    dstGm_.SetGlobalBuffer((__gm__ T *)(dst + tilingData_->dstStorageOffset * sizeof(T)));
    srcGm_.SetGlobalBuffer((__gm__ T *)(src + tilingData_->srcStorageOffset * sizeof(T)));

    blockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_->bufferSize);
    pipe_.InitBuffer(Dim4paramMagicBuf_, DIM4_PARAM_NUM * sizeof(uint32_t));
    pipe_.InitBuffer(Dim4paramShiftBuf_, DIM4_PARAM_NUM * sizeof(uint32_t));
    pipe_.InitBuffer(Dim5paramSrcStrideBuf_, DIM5_PARAM_NUM * sizeof(uint32_t));
    pipe_.InitBuffer(Dim5paramDstStrideBuf_, DIM5_PARAM_NUM * sizeof(uint64_t));
    for (int32_t x = DIM0_INDEX; x >= DIM4_INDEX; x--) {
        batchSize_ *= tilingData_->ubDstSize[x];
        tailBatchSize_ *= this->tailUbDstSize_[x];
    }
    int32_t batchSize_ = 1;
    int32_t tailBatchSize_ = 1;

    dim4UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX] * tilingData_->ubDstSize[DIM3_INDEX];
    dim3UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX];
    dim2UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX];
    dim1UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX];
    GetUintDivMagicAndShift<uint32_t>(m_[0], shift_[0], dim4UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[1], shift_[1], dim3UbdstStride_); 
    GetUintDivMagicAndShift<uint32_t>(m_[2], shift_[2], dim2UbdstStride_);   
    GetUintDivMagicAndShift<uint32_t>(m_[3], shift_[3], dim1UbdstStride_); 
}

template <typename T>
__aicore__ inline void ViewCopySimtDim5<T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    int64_t loopSize = tilingData_->blockFactor;
    if (blockOffset_ >= tilingData_->fusedBlockDims) {
        return;
    } else if ((blockOffset_ + tilingData_->blockFactor) > tilingData_->fusedBlockDims) {
        loopSize = tilingData_->fusedBlockDims - blockOffset_;
    }
    InitCopyParams();
    int64_t globalLoopIdx = 0;
    int64_t globalLoopIdxModUo = 0;
    LocalTensor<uint32_t> ParamDim4MagicUb = Dim4paramMagicBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim4ShiftUb = Dim4paramShiftBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim5SrcStrideUb = Dim5paramSrcStrideBuf_.Get<uint32_t>();
    for (uint32_t i = 0; i < MS_DIM4_IDX_NUM; i++) {
        ParamDim4MagicUb.SetValue(i, m_[i]);
        ParamDim4ShiftUb.SetValue(i, shift_[i]);
    }
    for (uint32_t i = 0; i < DIM5_IDX_NUM; i++) {
        ParamDim5SrcStrideUb.SetValue(i, tilingData_->contiguousUbDstStride[DIM8_IDX_NUM - i - 1]);
    }
    if (tilingData_->enableDstInt64 != 0) {
        LocalTensor<uint64_t> ParamDim5DstStrideUb = Dim5paramDstStrideBuf_.Get<uint64_t>();
        for (uint32_t i = 0; i < DIM5_IDX_NUM; i++) {
            ParamDim5DstStrideUb.SetValue(i, static_cast<uint64_t>(tilingData_->ubDstStride[DIM8_IDX_NUM - i - 1]));
        }
    } else {
        LocalTensor<uint32_t> ParamDim5DstStrideUb = Dim5paramDstStrideBuf_.Get<uint32_t>();
        for (uint32_t i = 0; i < DIM5_IDX_NUM; i++) {
            ParamDim5DstStrideUb.SetValue(i, static_cast<uint32_t>(tilingData_->ubDstStride[DIM8_IDX_NUM - i - 1]));
        }
    }
    DataSyncBarrier<MemDsbT::UB>();
    for (int64_t idx = 0; idx < loopSize; idx++) {
        globalLoopIdx = blockOffset_ + idx;
        globalLoopIdxModUo = globalLoopIdx % tilingData_->uo;
        if ((globalLoopIdxModUo) * tilingData_->ubFactor <= tilingData_->ubDimSize) {
            if ((globalLoopIdxModUo + 1) * tilingData_->ubFactor > tilingData_->ubDimSize) {
                ProcessPerLoop(globalLoopIdx, tailDmaParam_, tailBatchSize_);
            } else {
                ProcessPerLoop(globalLoopIdx, dmaParam_, batchSize_);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopySimtDim5<T>::CopyIn(const GlobalTensor<T> &src,
    const MultiCopyParams<T, DIM5> &dmaParams)
{
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    this->CopyInDim5(src, srcLocal, dmaParams, tilingData_->enableMovAlign != 0);
    inQueue_.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void ViewCopySimtDim5<T>::ProcessPerLoop(int64_t globalLoopIdx,
    const MultiCopyParams<T, DIM5> &dmaParam, int32_t batchSize)
{
    LocalTensor<uint32_t> ParamDim4MagicUb = Dim4paramMagicBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim4ShiftUb = Dim4paramShiftBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim5SrcStrideUb = Dim5paramSrcStrideBuf_.Get<uint32_t>();
    int64_t srcOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockSrcStride,
                                          tilingData_->blockFusedDimsNumber);
    CopyIn(srcGm_[srcOffset], dmaParam);
    int64_t dstOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockDstStride,
                                          tilingData_->blockFusedDimsNumber);
    LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
    __ubuf__ T* srcAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    __gm__ T* dstAddr = (__gm__ T*)(dstGm_.GetPhyAddr()) + dstOffset;
    if (tilingData_->enableDstInt64 != 0) {
        LocalTensor<uint64_t> ParamDim5DstStrideUb = Dim5paramDstStrideBuf_.Get<uint64_t>();
        asc_vf_call<CopyUbToGmDim5<T, uint64_t>>(dim3(THREAD_DIM), dstAddr, srcAddr, batchSize,
            dim4UbdstStride_, dim3UbdstStride_, dim2UbdstStride_, dim1UbdstStride_, 
            (__ubuf__ uint32_t*)(ParamDim4MagicUb.GetPhyAddr()), (__ubuf__ uint32_t*)(ParamDim4ShiftUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim5SrcStrideUb.GetPhyAddr()),
            (__ubuf__ uint64_t*)(ParamDim5DstStrideUb.GetPhyAddr()));
    } else {
        LocalTensor<uint32_t> ParamDim5DstStrideUb = Dim5paramDstStrideBuf_.Get<uint32_t>();
        asc_vf_call<CopyUbToGmDim5<T, uint32_t>>(dim3(THREAD_DIM), dstAddr, srcAddr, batchSize,
            dim4UbdstStride_, dim3UbdstStride_, dim2UbdstStride_, dim1UbdstStride_,
            (__ubuf__ uint32_t*)(ParamDim4MagicUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim4ShiftUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim5SrcStrideUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim5DstStrideUb.GetPhyAddr()));
    }
    inQueue_.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void ViewCopySimtDim5<T>::InitCopyParams()
{
    dmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM2_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM3_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM4_INDEX]) },
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM4_INDEX])},
            {static_cast<uint32_t>(tilingData_->nddmaSize[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM4_INDEX])},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0}
        },
        0
    };

    tailDmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM2_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM3_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM4_INDEX])},
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM4_INDEX])},
            {static_cast<uint32_t>(this->tailNddmaSize_[DIM0_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM1_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM2_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM3_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM4_INDEX])},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0}
        },
        0
    };
}
}  // namespace ViewCopy

#endif  // VIEW_COPY_SIMT_DIM5_H_