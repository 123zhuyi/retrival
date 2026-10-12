/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPACE_TO_BATCH_ND_H
#define SPACE_TO_BATCH_ND_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "space_to_batch_nd_tiling_data.h"

using namespace AscendC;
using namespace Ops::Base;

constexpr static uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

template <typename T, int TilingKey>
class SpaceToBatchND {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SpaceToBatchNDTilingData* tilingData)
    {
        td_ = tilingData;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(ubBuf_, td_->bufferSize * 2);

        rank_ = td_->rank;
        N_ = td_->numSpatialDims;
        innerSize_ = td_->outShape[rank_ - 1];
        batchMul_ = 1;
        for (int i = 0; i < N_; i++) {
            bs_[i] = td_->blockShape[i];
            padTop_[i] = td_->padTop[i];
            spatial_[i] = td_->inShape[i + 1];
            padded_[i] = td_->inShape[i + 1] + td_->padTop[i] + td_->padBottom[i];
            batchMul_ *= bs_[i];
        }

        for (int ax = 0; ax < rank_; ax++) {
            layout_.axes[ax].outStride = 1;
            for (int k = ax + 1; k < rank_; k++) {
                layout_.axes[ax].outStride *= td_->outShape[k];
            }
            layout_.axes[ax].inStride = 1;
            for (int k = ax + 1; k < rank_; k++) {
                layout_.axes[ax].inStride *= td_->inShape[k];
            }
        }

        int64_t cAlignedBytes = CeilAlign(innerSize_ * static_cast<int64_t>(sizeof(T)),
                                          static_cast<int64_t>(UB_BLOCK_BYTES));
        cAligned_ = cAlignedBytes / sizeof(T);
        cAlignedBytes_ = cAlignedBytes;
    }

    __aicore__ inline void Process()
    {
        uint64_t blockIdx = GetBlockIdx();
        uint64_t startBlock = blockIdx * td_->perCoreCount;
        if (startBlock >= td_->totalCount) {
            return;
        }
        uint64_t endBlock = (startBlock + td_->perCoreCount < td_->totalCount) ? (startBlock + td_->perCoreCount) :
                                                                                 td_->totalCount;

        int64_t ubAxis = td_->ubAxis;

        for (uint64_t idx = startBlock; idx < endBlock; idx++) {
            uint64_t bufOff = (idx % 2) * (td_->bufferSize / sizeof(T));

            ComputeOutIndex(idx);

            int64_t tileStart = layout_.axes[ubAxis].start;
            int64_t tileCount = td_->ubFactor;
            if (tileStart + tileCount > td_->outShape[ubAxis]) {
                tileCount = td_->outShape[ubAxis] - tileStart;
            }

            int64_t dmaRows = tileCount;
            for (int64_t i = ubAxis + 1; i < rank_ - 1; i++) {
                dmaRows *= td_->outShape[i];
            }

            bool hasPadding = false;
            for (int64_t i = 0; i < N_; i++) {
                if (padded_[i] != spatial_[i]) {
                    hasPadding = true;
                    break;
                }
            }
            if (hasPadding) {
                Duplicate<T>(ubBuf_.Get<T>()[bufOff], T(0), td_->bufferSize / sizeof(T));
            }
            InsertSync(HardEvent::V_MTE2);

            int64_t outOff = 0;
            for (int64_t i = 0; i < rank_; i++) {
                outOff += layout_.axes[i].start * layout_.axes[i].outStride;
            }

            if constexpr (TilingKey == 1) {
                CopyInKey1(bufOff, tileStart, tileCount);
            } else if constexpr (TilingKey == 2) {
                CopyInKey2(bufOff, tileStart, tileCount);
            } else if constexpr (TilingKey == 3) {
                CopyInKey3(bufOff, tileStart, tileCount);
            } else if constexpr (TilingKey == 4) {
                CopyInKey4(bufOff, tileStart, tileCount, td_->ubAxis);
            } else {
                CopyInKey5Plus(bufOff, tileStart, tileCount);
            }
            InsertSync(HardEvent::V_MTE3);
            InsertSync(HardEvent::MTE2_MTE3);
            CopyOut(bufOff, tileCount, outOff);
            InsertSync(HardEvent::MTE3_V);
            InsertSync(HardEvent::MTE3_MTE2);
        }
    }

private:
    static constexpr int64_t MAX_RANK = ::MAX_RANK;
    static constexpr int64_t MAX_SP = ::MAX_SPATIAL;

    struct AxisLayout {
        int64_t start;
        int64_t length;
        int64_t inStride;
        int64_t outStride;
    };
    struct Layout {
        AxisLayout axes[MAX_RANK];
        int64_t inOffset;
        int64_t outOffset;
    };

    __aicore__ inline void ComputeOutIndex(int64_t tileIdx)
    {
        int64_t tmp = tileIdx;
        int64_t ubAxis = td_->ubAxis;
        int64_t ubFactor = td_->ubFactor;

        for (int64_t axis = rank_ - 1; axis >= 0; --axis) {
            if (axis > ubAxis) {
                layout_.axes[axis].start = 0;
                continue;
            }
            int64_t dim = td_->outShape[axis];
            if (axis == ubAxis) {
                dim = (dim + ubFactor - 1) / ubFactor;
                layout_.axes[axis].start = (tmp % dim) * ubFactor;
            } else {
                layout_.axes[axis].start = tmp % dim;
            }
            tmp /= dim;
        }

        DecomposeBatchIndex(layout_.axes[0].start);
    }

    __aicore__ inline void DecomposeBatchIndex(int64_t outBatchIdx)
    {
        int64_t batch = td_->inShape[0];
        int64_t batchIdx = outBatchIdx % batch;
        int64_t blockLinear = outBatchIdx / batch;
        baseInOff_ = batchIdx * layout_.axes[0].inStride;
        baseOutOff_ = outBatchIdx * layout_.axes[0].outStride;
        layout_.axes[0].start = outBatchIdx;

        int64_t rem = blockLinear;
        for (int64_t i = N_ - 1; i >= 0; i--) {
            blockIndices_[i] = rem % bs_[i];
            rem /= bs_[i];
            baseInOff_ += (blockIndices_[i] - padTop_[i]) * layout_.axes[i + 1].inStride;
        }
    }

    __aicore__ inline bool IsValid()
    {
        for (int64_t i = 0; i < N_; i++) {
            int64_t pc = layout_.axes[i + 1].start * bs_[i] + blockIndices_[i];
            int64_t ic = pc - padTop_[i];
            if (ic < 0 || ic >= spatial_[i]) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool HasValidRange(int64_t axis)
    {
        int64_t bi = blockIndices_[axis - 1];
        int64_t pt = padTop_[axis - 1];
        int64_t sp = spatial_[axis - 1];
        int64_t bsVal = bs_[axis - 1];
        int64_t absMin = (pt > bi) ? ((pt - bi + bsVal - 1) / bsVal) : 0;
        int64_t absMax = (pt + sp - 1 - bi) / bsVal;
        return absMin <= absMax;
    }

    __aicore__ inline bool CheckOuterValid(int64_t skipDim)
    {
        for (int64_t i = 1; i <= N_; i++) {
            if (i == skipDim) {
                continue;
            }
            if (!HasValidRange(i)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool CheckOuterValid(int64_t skipDim1, int64_t skipDim2)
    {
        for (int64_t i = 1; i <= N_; i++) {
            if (i == skipDim1 || i == skipDim2) {
                continue;
            }
            if (!HasValidRange(i)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void CalcValidRange(int64_t axis, int64_t tileStart, int64_t tileCount, int64_t& validStart,
                                          int64_t& validCount)
    {
        int64_t bi = blockIndices_[axis - 1];
        int64_t bsVal = bs_[axis - 1];
        int64_t pt = padTop_[axis - 1];
        int64_t sp = spatial_[axis - 1];
        int64_t absMin = (pt > bi) ? ((pt - bi + bsVal - 1) / bsVal) : 0;
        int64_t absMax = (pt + sp - 1 - bi) / bsVal;
        validStart = (absMin > tileStart) ? absMin : tileStart;
        int64_t validEnd = (absMax < tileStart + tileCount - 1) ? absMax : (tileStart + tileCount - 1);
        validCount = (validEnd >= validStart) ? (validEnd - validStart + 1) : 0;
    }

    __aicore__ inline void CalcBlockCountRange(int64_t fullSize, int64_t& bcStart, int64_t& bcCount)
    {
        bcStart = 0;
        bcCount = fullSize;
        if (padded_[N_ - 1] != spatial_[N_ - 1]) {
            int64_t biN = blockIndices_[N_ - 1];
            int64_t bsN = bs_[N_ - 1];
            int64_t ptN = padTop_[N_ - 1];
            int64_t spN = spatial_[N_ - 1];
            bcStart = (ptN > biN) ? ((ptN - biN + bsN - 1) / bsN) : 0;
            int64_t bcEnd = (ptN + spN - 1 - biN) / bsN + 1;
            if (bcEnd > fullSize) {
                bcEnd = fullSize;
            }
            if (bcStart > bcEnd) {
                bcStart = bcEnd;
            }
            bcCount = bcEnd - bcStart;
            layout_.axes[N_].start = bcStart;
        }
    }

    __aicore__ inline int64_t CalcInOffset(int64_t innerPos = 0)
    {
        int64_t offset = baseInOff_;
        for (int64_t i = 0; i < N_; i++) {
            offset += layout_.axes[i + 1].start * bs_[i] * layout_.axes[i + 1].inStride;
        }
        return offset + innerPos;
    }

    __aicore__ inline int64_t CalcOutOffset(int64_t innerPos = 0)
    {
        int64_t offset = baseOutOff_;
        for (int64_t i = 1; i <= N_; i++) {
            offset += layout_.axes[i].start * layout_.axes[i].outStride;
        }
        return offset + innerPos;
    }

    __aicore__ inline void CopyIn(uint64_t bufOff, int64_t inOff, uint32_t blockLen, uint16_t blockCount,
                                  uint32_t srcStride, uint32_t dstStride)
    {
        DataCopyExtParams cp{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<T> pp{false, 0, 0, 0};
        DataCopyPad(ubBuf_.Get<T>()[bufOff], xGm_[inOff], cp, pp);
    }

    __aicore__ inline void InsertSync(const HardEvent& event)
    {
        event_t eventID = static_cast<event_t>(pipe_.FetchEventID(event));
        switch (event) {
            case HardEvent::MTE2_MTE3:
                SetFlag<HardEvent::MTE2_MTE3>(eventID);
                WaitFlag<HardEvent::MTE2_MTE3>(eventID);
                break;
            case HardEvent::MTE3_MTE2:
                SetFlag<HardEvent::MTE3_MTE2>(eventID);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID);
                break;
            case HardEvent::V_MTE2:
                SetFlag<HardEvent::V_MTE2>(eventID);
                WaitFlag<HardEvent::V_MTE2>(eventID);
                break;
            case HardEvent::MTE3_V:
                SetFlag<HardEvent::MTE3_V>(eventID);
                WaitFlag<HardEvent::MTE3_V>(eventID);
                break;
            case HardEvent::V_MTE3:
                SetFlag<HardEvent::V_MTE3>(eventID);
                WaitFlag<HardEvent::V_MTE3>(eventID);
                break;
            default:
                break;
        }
    }

    __aicore__ inline void CopyInBatchLoop(uint64_t bufOff, int64_t tileStart, int64_t tileCount)
    {
        uint32_t cAlignedGap = static_cast<uint32_t>((cAligned_ - innerSize_) * sizeof(T) / UB_BLOCK_BYTES);
        uint32_t blockLen = static_cast<uint32_t>(innerSize_ * sizeof(T));
        uint64_t bufOffset = bufOff;
        for (int64_t t = 0; t < tileCount; t++) {
            DecomposeBatchIndex(tileStart + t);
            if (IsValid()) {
                CopyIn(bufOffset, CalcInOffset(), blockLen, 1, 0, cAlignedGap);
            }
            bufOffset += cAligned_;
        }
    }

    __aicore__ inline void CopyInKey1(uint64_t bufOff, int64_t tileStart, int64_t tileCount)
    {
        int64_t ubAxis = td_->ubAxis;
        int64_t innerStart = layout_.axes[ubAxis].start;
        if (IsValid()) {
            CopyIn(bufOff, CalcInOffset(innerStart), tileCount * sizeof(T), 1, 0, 0);
        }
    }

    __aicore__ inline void CopyInKey2(uint64_t bufOff, int64_t tileStart, int64_t tileCount)
    {
        int64_t ubAxis = td_->ubAxis;
        uint32_t cAlignedGap = static_cast<uint32_t>((cAligned_ - innerSize_) * sizeof(T) / UB_BLOCK_BYTES);
        uint32_t blockLen = static_cast<uint32_t>(innerSize_ * sizeof(T));

        int64_t validStart = tileStart, validCount = tileCount;
        if (!CheckOuterValid(ubAxis)) {
            validCount = 0;
        } else {
            for (int64_t i = 1; i <= N_; i++) {
                if (i == ubAxis)
                    continue;
                int64_t inPos = layout_.axes[i].start * bs_[i - 1] + blockIndices_[i - 1] - padTop_[i - 1];
                if (inPos < 0 || inPos >= spatial_[i - 1]) {
                    validCount = 0;
                    break;
                }
            }
        }
        if (validCount > 0 && padded_[ubAxis - 1] != spatial_[ubAxis - 1]) {
            CalcValidRange(ubAxis, tileStart, tileCount, validStart, validCount);
        }

        if (validCount > 0) {
            layout_.axes[ubAxis].start = validStart;
            uint32_t srcGap = static_cast<uint32_t>((bs_[N_ - 1] * layout_.axes[N_].inStride - innerSize_) * sizeof(T));
            CopyIn(bufOff + (validStart - tileStart) * cAligned_, CalcInOffset(), blockLen, validCount, srcGap,
                   cAlignedGap);
        }
    }

    __aicore__ inline void CopyInKey3(uint64_t bufOff, int64_t tileStart, int64_t tileCount)
    {
        int64_t ubAxis = td_->ubAxis;
        if (ubAxis == 0) {
            CopyInBatchLoop(bufOff, tileStart, tileCount);
            return;
        }

        if (!CheckOuterValid(ubAxis)) {
            return;
        }

        int64_t validStart, validTileCount;
        CalcValidRange(ubAxis, tileStart, tileCount, validStart, validTileCount);
        if (validTileCount <= 0) {
            return;
        }
        layout_.axes[ubAxis].start = validStart;

        int64_t bcFullSize = td_->outShape[N_];
        int64_t bcStart, bcCount;
        CalcBlockCountRange(bcFullSize, bcStart, bcCount);
        if (bcCount <= 0) {
            return;
        }

        uint32_t cAlignedGap = static_cast<uint32_t>((cAligned_ - innerSize_) * sizeof(T) / UB_BLOCK_BYTES);
        uint32_t blockLen = static_cast<uint32_t>(innerSize_ * sizeof(T));
        uint32_t srcGap = static_cast<uint32_t>((bs_[N_ - 1] * layout_.axes[N_].inStride - innerSize_) * sizeof(T));
        int64_t bsTa = bs_[ubAxis - 1];

        LoopModeParams loopParams;
        loopParams.loop1Size = validTileCount;
        loopParams.loop1SrcStride = static_cast<uint32_t>(bsTa * layout_.axes[ubAxis].inStride * sizeof(T));
        loopParams.loop1DstStride = static_cast<uint32_t>(bcFullSize * cAlignedBytes_);
        loopParams.loop2Size = 1;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        CopyIn(bufOff + (validStart - tileStart) * bcFullSize * cAligned_ + bcStart * cAligned_, CalcInOffset(),
               blockLen, static_cast<uint16_t>(bcCount), srcGap, cAlignedGap);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void CopyInKey4(uint64_t bufOff, int64_t tileStart, int64_t tileCount, int64_t ubAxis)
    {
        if (ubAxis == 0) {
            CopyInBatchLoop(bufOff, tileStart, tileCount);
            return;
        }

        if (!CheckOuterValid(ubAxis, ubAxis + 1)) {
            return;
        }

        int64_t validStart, validTileCount;
        CalcValidRange(ubAxis, tileStart, tileCount, validStart, validTileCount);
        if (validTileCount <= 0) {
            return;
        }
        layout_.axes[ubAxis].start = validStart;

        int64_t bcFullSize = td_->outShape[N_];
        int64_t bcStart, bcCount;
        CalcBlockCountRange(bcFullSize, bcStart, bcCount);
        if (bcCount <= 0) {
            return;
        }

        uint32_t cAlignedGap = static_cast<uint32_t>((cAligned_ - innerSize_) * sizeof(T) / UB_BLOCK_BYTES);
        uint32_t blockLen = static_cast<uint32_t>(innerSize_ * sizeof(T));
        uint32_t srcGap = static_cast<uint32_t>((bs_[N_ - 1] * layout_.axes[N_].inStride - innerSize_) * sizeof(T));
        int64_t bsTa = bs_[ubAxis - 1];
        int64_t loop1Size = (ubAxis + 1 < N_) ? td_->outShape[ubAxis + 1] : 1;

        LoopModeParams loopParams;
        loopParams.loop1Size = loop1Size;
        loopParams.loop1SrcStride = (ubAxis + 1 < N_) ?
                                        static_cast<uint32_t>(bs_[ubAxis] * layout_.axes[ubAxis + 1].inStride *
                                                              sizeof(T)) :
                                        0;
        loopParams.loop1DstStride = static_cast<uint32_t>(bcFullSize * cAlignedBytes_);
        loopParams.loop2Size = validTileCount;
        loopParams.loop2SrcStride = static_cast<uint32_t>(bsTa * layout_.axes[ubAxis].inStride * sizeof(T));
        loopParams.loop2DstStride = static_cast<uint32_t>(loop1Size * bcFullSize * cAlignedBytes_);

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        CopyIn(bufOff + (validStart - tileStart) * loop1Size * bcFullSize * cAligned_ + bcStart * cAligned_,
               CalcInOffset(), blockLen, static_cast<uint16_t>(bcCount), srcGap, cAlignedGap);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void CopyInKey5Plus(uint64_t bufOff, int64_t tileStart, int64_t tileCount)
    {
        int64_t key4UbAxis = rank_ - 4;
        int64_t key4Size = td_->outShape[key4UbAxis] * cAligned_;
        for (int64_t i = key4UbAxis + 1; i < rank_ - 1; i++) {
            key4Size *= td_->outShape[i];
        }

        int64_t total = tileCount;
        for (int64_t i = td_->ubAxis + 1; i < key4UbAxis; i++) {
            total *= td_->outShape[i];
        }

        uint64_t baseOff = bufOff;
        for (int64_t idx = 0; idx < total; idx++) {
            int64_t tmp = idx;
            for (int64_t i = key4UbAxis - 1; i >= td_->ubAxis; i--) {
                int64_t dim = (i == td_->ubAxis) ? tileCount : td_->outShape[i];
                layout_.axes[i].start = tmp % dim;
                tmp /= dim;
            }
            if (td_->ubAxis == key4UbAxis - 1) {
                layout_.axes[td_->ubAxis].start += tileStart;
            }
            CopyInKey4(baseOff, 0, td_->outShape[key4UbAxis], key4UbAxis);
            baseOff += key4Size;
        }
    }

    __aicore__ inline void CopyOut(uint64_t bufOff, int64_t tileCount, int64_t outOff)
    {
        if constexpr (TilingKey == 1) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(tileCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outOff], ubBuf_.Get<T>()[bufOff], cp);
        } else {
            uint32_t cAlignedGap = static_cast<uint32_t>((cAligned_ - innerSize_) * sizeof(T) / UB_BLOCK_BYTES);
            uint32_t blockLen = static_cast<uint32_t>(innerSize_ * sizeof(T));
            int64_t ubAxis = td_->ubAxis;
            int64_t blockCount = tileCount * (layout_.axes[ubAxis].outStride / innerSize_);

            DataCopyExtParams cp{static_cast<uint16_t>(blockCount), blockLen, cAlignedGap, 0, 0};
            DataCopyPad(yGm_[outOff], ubBuf_.Get<T>()[bufOff], cp);
        }
    }

    const SpaceToBatchNDTilingData* td_;
    Layout layout_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t bs_[MAX_SP];
    int64_t padTop_[MAX_SP];
    int64_t spatial_[MAX_SP];
    int64_t padded_[MAX_SP];
    int64_t blockIndices_[MAX_SP];
    int64_t baseInOff_;
    int64_t baseOutOff_;
    int64_t batchMul_;
    int64_t innerSize_;
    int64_t N_;
    int64_t rank_;
    int64_t cAligned_;
    int64_t cAlignedBytes_;
};

#endif
