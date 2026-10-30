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
 * \file view_copy_simt_dim8.h
 * \brief
 */

#ifndef VIEW_COPY_SIMT_DIM8_H_
#define VIEW_COPY_SIMT_DIM8_H_

#include "view_copy_base.h"

namespace ViewCopy {
using namespace AscendC;

template <typename T, typename U>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void CopyUbToGmDim8(__gm__ T* dst, __ubuf__ T* src, uint32_t batchSize,
    uint32_t ubDim7Stride, uint32_t ubDim6Stride, uint32_t ubDim5Stride, uint32_t ubDim4Stride, uint32_t ubDim3Stride,
    uint32_t ubDim2Stride, uint32_t ubDim1Stride, __ubuf__ uint32_t* ParamDim7MagicUb,
    __ubuf__ uint32_t* ParamDim7ShiftUb, __ubuf__ uint32_t* ParamDim8SrcStrideUb, __ubuf__ U* ParamDim8DstStrideUb)
{
    for (uint32_t i = threadIdx.x; i < batchSize; i += blockDim.x) {
        uint32_t idx = i;
        uint32_t srcIndex = 0;
        U dstIndex = 0;

        uint32_t t7 = __umulhi(idx, ParamDim7MagicUb[DIM0]);
        t7 = t7 + idx;
        uint32_t dim7Index = t7 >> ParamDim7ShiftUb[DIM0];
        uint32_t newIdx7 = idx - dim7Index * ubDim7Stride;
        srcIndex += (dim7Index * ParamDim8SrcStrideUb[DIM7]);
        dstIndex += (dim7Index * ParamDim8DstStrideUb[DIM7]);

        uint32_t t6 = __umulhi(newIdx7, ParamDim7MagicUb[DIM1]);
        t6 = t6 + newIdx7;
        uint32_t dim6Index = t6 >> ParamDim7ShiftUb[DIM1];
        uint32_t newIdx6 = newIdx7 - dim6Index * ubDim6Stride;
        srcIndex += (dim6Index * ParamDim8SrcStrideUb[DIM6]);
        dstIndex += (dim6Index * ParamDim8DstStrideUb[DIM6]);

        uint32_t t5 = __umulhi(newIdx6, ParamDim7MagicUb[DIM2]);
        t5 = t5 + newIdx6;
        uint32_t dim5Index = t5 >> ParamDim7ShiftUb[DIM2];
        uint32_t newIdx5 = newIdx6 - dim5Index * ubDim5Stride;
        srcIndex += (dim5Index * ParamDim8SrcStrideUb[DIM5]);
        dstIndex += (dim5Index * ParamDim8DstStrideUb[DIM5]);

        uint32_t t4 = __umulhi(newIdx5, ParamDim7MagicUb[DIM3]);
        t4 = t4 + newIdx5;
        uint32_t dim4Index = t4 >> ParamDim7ShiftUb[DIM3];
        uint32_t newIdx4 = newIdx5 - dim4Index * ubDim4Stride;
        srcIndex += (dim4Index * ParamDim8SrcStrideUb[DIM4]);
        dstIndex += (dim4Index * ParamDim8DstStrideUb[DIM4]);

        uint32_t t3 = __umulhi(newIdx4, ParamDim7MagicUb[DIM4]);
        t3 = t3 + newIdx4;
        uint32_t dim3Index = t3 >> ParamDim7ShiftUb[DIM4];
        uint32_t newIdx3 = newIdx4 - dim3Index * ubDim3Stride;
        srcIndex += (dim3Index * ParamDim8SrcStrideUb[DIM3]);
        dstIndex += (dim3Index * ParamDim8DstStrideUb[DIM3]);

        uint32_t t2 = __umulhi(newIdx3, ParamDim7MagicUb[DIM5]);
        t2 = t2 + newIdx3;
        uint32_t dim2Index = t2 >> ParamDim7ShiftUb[DIM5];
        uint32_t newIdx2 = newIdx3 - dim2Index * ubDim2Stride;
        srcIndex += (dim2Index * ParamDim8SrcStrideUb[DIM2]);
        dstIndex += (dim2Index * ParamDim8DstStrideUb[DIM2]);

        uint32_t t1 = __umulhi(newIdx2, ParamDim7MagicUb[DIM6]);
        t1 = t1 + newIdx2;
        uint32_t dim1Index = t1 >> ParamDim7ShiftUb[DIM6];
        uint32_t dim0Index = newIdx2 - dim1Index * ubDim1Stride;
        srcIndex += (dim1Index * ParamDim8SrcStrideUb[DIM1]);
        dstIndex += (dim1Index * ParamDim8DstStrideUb[DIM1]);

        srcIndex += (dim0Index * ParamDim8SrcStrideUb[DIM0]);
        dstIndex += (dim0Index * ParamDim8DstStrideUb[DIM0]);
        dst[dstIndex] = src[srcIndex];
    }
}

template <typename T> class ViewCopySimtDim8 : public ViewCopyBase<T> {
public:
    __aicore__ inline ViewCopySimtDim8(TPipe &pipe, const ViewCopyTilingData *tilingData) :
        pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset, GM_ADDR src,
        GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset, GM_ADDR out);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void CustomNddma(LocalTensor<T> &dst, const GlobalTensor<T> &src, const int32_t *loopSize,
        const int64_t *srcStride, const int64_t *dstStride, const MultiCopyParams<T, 5> &dmaParam);
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM5> &dmaParam,
        const int32_t *copyInLoopSize, int32_t batchSize);
private:
    TPipe &pipe_;
    const ViewCopyTilingData *tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;
    TBuf<TPosition::VECCALC> Dim7paramMagicBuf_;
    TBuf<TPosition::VECCALC> Dim7paramShiftBuf_;
    TBuf<TPosition::VECCALC> Dim8paramSrcStrideBuf_;
    TBuf<TPosition::VECCALC> Dim8paramDstStrideBuf_;

    MultiCopyParams<T, DIM5> tailDmaParam_;
    MultiCopyParams<T, DIM5> dmaParam_;
    DataCopyExtParams tailCopyParams_;
    DataCopyExtParams copyParams_;

    int64_t blockOffset_ = 0;
    int32_t batchSize_ = 1;
    int32_t tailBatchSize_ = 1;
    uint32_t dim7UbdstStride_ = 1;
    uint32_t dim6UbdstStride_ = 1;
    uint32_t dim5UbdstStride_ = 1;
    uint32_t dim4UbdstStride_ = 1;
    uint32_t dim3UbdstStride_ = 1;
    uint32_t dim2UbdstStride_ = 1;
    uint32_t dim1UbdstStride_ = 1;
    uint32_t shift_[7];
    uint32_t m_[7];
};

template <typename T>
__aicore__ inline void ViewCopySimtDim8<T>::Init(GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride,
    GM_ADDR dstStorageOffset, GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset, GM_ADDR out)
{
    this->ParseTilingData(tilingData_);
    srcGm_.SetGlobalBuffer((__gm__ T *)(src + tilingData_->srcStorageOffset * sizeof(T)));
    dstGm_.SetGlobalBuffer((__gm__ T *)(dst + tilingData_->dstStorageOffset * sizeof(T)));

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_->bufferSize);
    pipe_.InitBuffer(Dim7paramMagicBuf_, DIM7_PARAM_NUM * sizeof(uint32_t));
    pipe_.InitBuffer(Dim7paramShiftBuf_, DIM7_PARAM_NUM * sizeof(uint32_t));
    pipe_.InitBuffer(Dim8paramSrcStrideBuf_, DIM8_PARAM_NUM * sizeof(uint32_t));
    pipe_.InitBuffer(Dim8paramDstStrideBuf_, DIM8_PARAM_NUM * sizeof(uint64_t));
    blockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    for (int32_t x = DIM0_INDEX; x >= DIM7_INDEX; x--) {
        batchSize_ *= tilingData_->ubDstSize[x];
        tailBatchSize_ *= this->tailUbDstSize_[x];
    }

    dim7UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX] * tilingData_->ubDstSize[DIM3_INDEX] * tilingData_->ubDstSize[DIM4_INDEX] *
        tilingData_->ubDstSize[DIM5_INDEX] * tilingData_->ubDstSize[DIM6_INDEX];
    dim6UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX] * tilingData_->ubDstSize[DIM3_INDEX] * tilingData_->ubDstSize[DIM4_INDEX] *
        tilingData_->ubDstSize[DIM5_INDEX];
    dim5UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX] * tilingData_->ubDstSize[DIM3_INDEX] * tilingData_->ubDstSize[DIM4_INDEX];
    dim4UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX] * tilingData_->ubDstSize[DIM3_INDEX];
    dim3UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX] *
        tilingData_->ubDstSize[DIM2_INDEX];
    dim2UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX];
    dim1UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX];
    GetUintDivMagicAndShift<uint32_t>(m_[0], shift_[0], dim7UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[1], shift_[1], dim6UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[2], shift_[2], dim5UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[3], shift_[3], dim4UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[4], shift_[4], dim3UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[5], shift_[5], dim2UbdstStride_);
    GetUintDivMagicAndShift<uint32_t>(m_[6], shift_[6], dim1UbdstStride_);
}

template <typename T> __aicore__ inline void ViewCopySimtDim8<T>::Process()
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
    LocalTensor<uint32_t> ParamDim7MagicUb = Dim7paramMagicBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim7ShiftUb = Dim7paramShiftBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim8SrcStrideUb = Dim8paramSrcStrideBuf_.Get<uint32_t>();
    for (uint32_t i = 0; i < MS_IDX_NUM; i++) {
        ParamDim7MagicUb.SetValue(i, m_[i]);
        ParamDim7ShiftUb.SetValue(i, shift_[i]);
    }
    for (uint32_t i = 0; i < DIM8_IDX_NUM; i++) {
        ParamDim8SrcStrideUb.SetValue(i, tilingData_->contiguousUbDstStride[DIM8_IDX_NUM - i - 1]);
    }
    if (tilingData_->enableDstInt64 != 0) {
        LocalTensor<uint64_t> ParamDim8DstStrideUb = Dim8paramDstStrideBuf_.Get<uint64_t>();
        for (uint32_t i = 0; i < DIM8_IDX_NUM; i++) {
            ParamDim8DstStrideUb.SetValue(i, static_cast<uint64_t>(tilingData_->ubDstStride[DIM8_IDX_NUM - i - 1]));
        }
    } else {
        LocalTensor<uint32_t> ParamDim8DstStrideUb = Dim8paramDstStrideBuf_.Get<uint32_t>();
        for (uint32_t i = 0; i < DIM8_IDX_NUM; i++) {
            ParamDim8DstStrideUb.SetValue(i, static_cast<uint32_t>(tilingData_->ubDstStride[DIM8_IDX_NUM - i - 1]));
        }
    }
    DataSyncBarrier<MemDsbT::UB>();
    for (int64_t idx = 0; idx < loopSize; idx++) {
        globalLoopIdx = blockOffset_ + idx;
        globalLoopIdxModUo = globalLoopIdx % tilingData_->uo;
        if ((globalLoopIdxModUo) * tilingData_->ubFactor <= tilingData_->ubDimSize) {
            if ((globalLoopIdxModUo + 1) * tilingData_->ubFactor > tilingData_->ubDimSize) {
                ProcessPerLoop(globalLoopIdx, tailDmaParam_, this->tailNddmaSize_, tailBatchSize_);
            } else {
                ProcessPerLoop(globalLoopIdx, dmaParam_, tilingData_->nddmaSize, batchSize_);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopySimtDim8<T>::ProcessPerLoop(int64_t globalLoopIdx,
    const MultiCopyParams<T, DIM5> &dmaParam, const int32_t *copyInLoopSize, int32_t batchSize)
{
    LocalTensor<uint32_t> ParamDim7MagicUb = Dim7paramMagicBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim7ShiftUb = Dim7paramShiftBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> ParamDim8SrcStrideUb = Dim8paramSrcStrideBuf_.Get<uint32_t>();
    int64_t srcOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockSrcStride,
        tilingData_->blockFusedDimsNumber);
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    CustomNddma(srcLocal, srcGm_[srcOffset], copyInLoopSize, tilingData_->nddmaStride,
        tilingData_->contiguousUbSrcStride, dmaParam);
    inQueue_.EnQue(srcLocal);
    int64_t dstOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockDstStride,
        tilingData_->blockFusedDimsNumber);
    LocalTensor<T> dstLocal = inQueue_.DeQue<T>();

    __ubuf__ T* srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    __gm__ T* dstAddr = (__gm__ T*)(dstGm_.GetPhyAddr()) + dstOffset;
    if (tilingData_->enableDstInt64 != 0) {
        LocalTensor<uint64_t> ParamDim8DstStrideUb = Dim8paramDstStrideBuf_.Get<uint64_t>();
        asc_vf_call<CopyUbToGmDim8<T, uint64_t>>(dim3(THREAD_DIM), dstAddr, srcAddr, batchSize,
            dim7UbdstStride_, dim6UbdstStride_, dim5UbdstStride_, dim4UbdstStride_, dim3UbdstStride_, dim2UbdstStride_,
            dim1UbdstStride_, (__ubuf__ uint32_t*)(ParamDim7MagicUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim7ShiftUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim8SrcStrideUb.GetPhyAddr()),
            (__ubuf__ uint64_t*)(ParamDim8DstStrideUb.GetPhyAddr()));
    } else {
        LocalTensor<uint32_t> ParamDim8DstStrideUb = Dim8paramDstStrideBuf_.Get<uint32_t>();
        asc_vf_call<CopyUbToGmDim8<T, uint32_t>>(dim3(THREAD_DIM), dstAddr, srcAddr, batchSize,
            dim7UbdstStride_, dim6UbdstStride_, dim5UbdstStride_, dim4UbdstStride_, dim3UbdstStride_, dim2UbdstStride_,
            dim1UbdstStride_, (__ubuf__ uint32_t*)(ParamDim7MagicUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim7ShiftUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim8SrcStrideUb.GetPhyAddr()),
            (__ubuf__ uint32_t*)(ParamDim8DstStrideUb.GetPhyAddr()));
    }
    inQueue_.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void ViewCopySimtDim8<T>::CustomNddma(LocalTensor<T> &dst, const GlobalTensor<T> &src,
    const int32_t *loopSize, const int64_t *srcStride, const int64_t *dstStride, const MultiCopyParams<T, 5> &dmaParam)
{
    // srcSize (dim7, dim6, dim5, dim4, dim3, dim2, dim1, dim0)
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;

    for (int64_t loopDim7 = 0; loopDim7 < loopSize[DIM7_INDEX]; loopDim7++) {
        for (int64_t loopDim6 = 0; loopDim6 < loopSize[DIM6_INDEX]; loopDim6++) {
            for (int64_t loopDim5 = 0; loopDim5 < loopSize[DIM5_INDEX]; loopDim5++) {
                srcOffset = loopDim7 * srcStride[DIM7_INDEX] + loopDim6 * srcStride[DIM6_INDEX] +
                    loopDim5 * srcStride[DIM5_INDEX];
                dstOffset = loopDim7 * dstStride[DIM7_INDEX] + loopDim6 * dstStride[DIM6_INDEX] +
                    loopDim5 * dstStride[DIM5_INDEX];
                DataCopy(dst[dstOffset], src[srcOffset], dmaParam);
            }
        }
    }
}

template <typename T> __aicore__ inline void ViewCopySimtDim8<T>::InitCopyParams()
{
    dmaParam_ = { 
        { 
            { static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM2_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM3_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM4_INDEX]) },
            { static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM4_INDEX]) },
            { static_cast<uint32_t>(tilingData_->nddmaSize[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM4_INDEX]) },
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 }
        },
        0
    };

    tailDmaParam_ = {
        {
            { static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM2_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM3_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM4_INDEX]) },
            { static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM4_INDEX]) },
            { static_cast<uint32_t>(this->tailNddmaSize_[DIM0_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM1_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM2_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM3_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM4_INDEX]) },
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 }
        },
        0
    };
}
} // namespace ViewCopy

#endif // VIEW_COPY_SIMT_DIM8_H_