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
 * \file attenmask_gs1.h
 * \brief
 */

#ifndef ATTENMASK_GS1_H
#define ATTENMASK_GS1_H

enum LAYOUT_Q {
    GS,
    SG,
    S1_EQUAL1,
};

enum MaskDataType : uint8_t {
    MASK_BOOL,
    MASK_INT8,
    MASK_UINT8,
    MASK_FP16,
};

enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

struct MaskInfo {
    uint32_t gs1StartIdx;
    uint32_t gs1dealNum;
    uint32_t s1Size;
    uint32_t gSize;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;

    int64_t preToken = 0;
    int64_t nextToken = 0;

    // for bss & bs
    uint32_t batchIdx;
    uint32_t attenMaskBatchStride;
    uint32_t attenMaskS1Stride;
    uint32_t attenMaskDstStride = 0;

    LAYOUT_Q layout;
    MaskDataType attenMaskType;
    SparseMode sparseMode;
    uint32_t maskValue;

    uint64_t s1LeftPaddingSize = 0;
    uint64_t s2LeftPaddingSize = 0;
};

__aicore__ inline uint64_t ComputeAttenMaskOffsetNoCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    uint64_t bOffset = static_cast<uint64_t>(info.batchIdx) * static_cast<uint64_t>(info.attenMaskBatchStride);
    uint64_t s1Offset = (info.s1LeftPaddingSize + s1StartIdx % info.s1Size) * info.attenMaskS1Stride;
    uint64_t s2Offset = info.s2LeftPaddingSize + info.s2StartIdx;
    return bOffset + s1Offset + s2Offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0; // sparse2 本身原点就是左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken = static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.sparseMode == BAND) { // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta :
            attenMaskSizeAlign) * info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressPre(MaskInfo &info, uint32_t s1StartIdx)
{
    // 统一以左上角为原点计算token
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) - static_cast<int64_t>(info.s2Size);
    int64_t delta = -preToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(info.s2StartIdx) - 1;
    uint64_t offset = 0;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta :
            attenMaskSizeAlign) * info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskInfo &info, uint32_t s1StartIdx = 0, bool isPre = false)
{
    if (isPre) {
        return ComputeAttenMaskOffsetCompressPre(info, s1StartIdx);
    } else {
        if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
            return ComputeAttenMaskOffsetNoCompress(info, s1StartIdx);
        } else {
            return ComputeAttenMaskOffsetCompress(info, s1StartIdx);
        }
    }
}

__aicore__ inline bool IsSkipAttentionmaskForPre(MaskInfo &info)
{
    if (info.sparseMode != BAND) {
        return true;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }
    // 统一以左上角为原点计算Token
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) - static_cast<int64_t>(info.s2Size);
    int32_t s1EndIdx = info.layout == GS ? s1StartIdx + info.gs1dealNum :
                       (info.gs1StartIdx + info.gs1dealNum) / info.gSize;

    if (static_cast<int64_t>(info.s2StartIdx) + preToken >= static_cast<int64_t>(s1EndIdx)) {
        return true;
    }
    return false;
}

__aicore__ inline bool IsSkipAttentionmask(MaskInfo &info)
{
    if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
        return false;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }

    int64_t nextToken = 0; // sparse2 本身原点就在左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken = static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算Token
    } else if (info.sparseMode == BAND) { // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr, MaskInfo &info,
                                             uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
{
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
    dataCopyParams.dstStride = info.attenMaskDstStride;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};

    DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline bool CheckIsSkipAttenMask(LocalTensor<T> &attenMaskUb, MaskInfo &info, bool isPre)
{
    if ((isPre && IsSkipAttentionmaskForPre(info)) || (!isPre && IsSkipAttentionmask(info))) {
        Duplicate(attenMaskUb, static_cast<T>(0U), info.gs1dealNum * Align(info.s2dealNum, 32U));
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
        PipeBarrier<PIPE_V>();
        return true;
    }
    return false;
}

template <typename T>
__aicore__ inline void AttentionmaskCopyInForGsLayout(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                      MaskInfo &info, bool isPre = false)
{
    if (CheckIsSkipAttenMask(attenMaskUb, info, isPre)) {
        return;
    }
    int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
    uint32_t attenMaskS2Stride = Align(info.s2dealNum, 32U) + 32 * info.attenMaskDstStride;
    if (info.gs1dealNum <= info.s1Size) {
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, info.s1Size, isPre);
            LocalTensor<T> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskS2Stride];
            AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, s1EndIdx, isPre);
        } else {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx, isPre);
        }
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
    } else {
        uint32_t headS1Count = info.s1Size - s1StartIdx;
        uint32_t remainRowCount = info.gs1dealNum - headS1Count;
        uint32_t midGCount = remainRowCount / info.s1Size;
        uint32_t tailS1Size = remainRowCount % info.s1Size;

        // 第一块完整的mask
        LocalTensor<T> attenMaskSecUb = attenMaskUb[headS1Count * attenMaskS2Stride];
        AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, info.s1Size, isPre);
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);

        // head
        DataCopy(attenMaskUb, attenMaskUb[info.s1Size * attenMaskS2Stride], headS1Count * attenMaskS2Stride);
        // mid
        for (uint32_t i = 1; i < midGCount; i++) {
            DataCopy(attenMaskUb[(headS1Count + i * info.s1Size) * attenMaskS2Stride],
                     attenMaskSecUb, info.s1Size * attenMaskS2Stride);
        }
        // tail
        if (tailS1Size > 0) {
            DataCopy(attenMaskUb[(headS1Count + midGCount * info.s1Size) * attenMaskS2Stride],
                     attenMaskSecUb, tailS1Size * attenMaskS2Stride);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                      MaskInfo &info, bool isPre = false)
{
    if (CheckIsSkipAttenMask(attenMaskUb, info, isPre)) {
        return;
    }
    uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
    uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t headGCount = s1Count > 1 ? (info.gSize - info.gs1StartIdx % info.gSize) : info.gs1dealNum;
    uint32_t remainRowCount = info.gs1dealNum - headGCount;
    uint32_t midS1Count = remainRowCount / info.gSize;
    uint32_t tailGSize = remainRowCount % info.gSize;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;

    // ub-head
    AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);

    // ub-remain
    if (remainRowCount > 0) {
        uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx + 1, isPre);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
        dataCopyParams.blockLen = info.s2dealNum;
        dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
        dataCopyParams.dstStride = (info.gSize - 1) * attenMaskS2Stride / 32 + info.attenMaskDstStride;
        // blockLen + padParams.len + dstStride = gSize * s2Size
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
        DataCopyPad(attenMaskUb[headGCount * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    }

    event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(enQueEvtID);
    WaitFlag<HardEvent::MTE2_V>(enQueEvtID);

    LocalTensor<int16_t> attenMaskUbDst = attenMaskUb.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
    uint32_t dstMaskOffset = attenMaskS2Stride / sizeof(int16_t);
    uint32_t srcMaskBaseOffset = 0;

    // head
    SetMaskCount();
    SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
        headGCount - 1, {1, 1, static_cast<uint16_t>(attenMaskS2Stride / 32), 0});
    dstMaskOffset += headGCount * attenMaskS2Stride / sizeof(int16_t);
    srcMaskBaseOffset += headGCount * attenMaskS2Stride / sizeof(int16_t);

    // mid
    for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
            info.gSize - 1, {1, 1, static_cast<uint16_t>(attenMaskS2Stride / 32), 0});
        dstMaskOffset += info.gSize * attenMaskS2Stride / sizeof(int16_t);
        srcMaskBaseOffset += info.gSize * attenMaskS2Stride / sizeof(int16_t);
    }
    // tail
    if (tailGSize > 0) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
            tailGSize - 1, {1, 1, static_cast<uint16_t>(attenMaskS2Stride / 32), 0});
    }
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AttentionmaskCopyGS1(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                            MaskInfo &info, bool isPre = false)
{
    if (info.layout == LAYOUT_Q::GS) {
        AttentionmaskCopyInForGsLayout(attenMaskUb, srcGmAddr, info, isPre);
    } else if (info.layout == LAYOUT_Q::SG) {
        AttentionmaskCopyInForSgLayout(attenMaskUb, srcGmAddr, info, isPre);
    }
}
#endif
