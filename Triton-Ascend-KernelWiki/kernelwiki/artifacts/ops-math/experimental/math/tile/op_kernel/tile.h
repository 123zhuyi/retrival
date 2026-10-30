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
 * \file tile.h
 * \brief
 */
#ifndef TILE_H
#define TILE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tile_tiling_data.h"
#include "tile_tiling_key.h"

#ifndef COMPLEX64_TYPE_DEFINED
#define COMPLEX64_TYPE_DEFINED
using complex64 = uint64_t;
#endif

namespace TileKernel {

using namespace AscendC;

constexpr int32_t MAX_DIM = 8;
constexpr int32_t DEFAULT_BUFFER_BYTES = 32 * 1024;
constexpr int32_t DATA_BLOCK_BYTES = 32;

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dst, const GlobalTensor<T>& src, int32_t count)
{
    DataCopyExtParams rp{1, static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(dst, src, rp, pp);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dst, const LocalTensor<T>& src, int32_t count)
{
    DataCopyExtParams wp{1, static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(dst, src, wp);
}

template <typename T>
__aicore__ inline void BuildInnerRow(
    LocalTensor<T>& outBuf, const LocalTensor<T>& inData, int32_t srcBase, int32_t dstBase, int32_t innerDim,
    int32_t innerMult)
{
    if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
        constexpr int32_t ALIGN_ELEMS = DATA_BLOCK_BYTES / static_cast<int32_t>(sizeof(T));
        bool srcAligned = (srcBase % ALIGN_ELEMS == 0);
        bool dstAligned = (dstBase % ALIGN_ELEMS == 0);
        bool dimAligned = (innerDim % ALIGN_ELEMS == 0);
        if (srcAligned && dstAligned && dimAligned) {
            const uint32_t srcShape[] = {static_cast<uint32_t>(innerDim), 1};
            const uint32_t dstShape[] = {static_cast<uint32_t>(innerDim), static_cast<uint32_t>(innerMult)};
            LocalTensor<T> srcSub = inData[srcBase];
            LocalTensor<T> dstSub = outBuf[dstBase];
            BroadCast<T, 2, 1>(dstSub, srcSub, dstShape, srcShape);
            return;
        }
    }
    if constexpr (sizeof(T) == 4) {
        if ((innerDim % 2) == 0) {
            auto outU64 = outBuf.template ReinterpretCast<uint64_t>();
            auto inU64 = inData.template ReinterpretCast<uint64_t>();
            int32_t hD = innerDim / 2;
            for (int32_t e = 0; e < hD; e++) {
                uint64_t val = inU64.GetValue(srcBase / 2 + e);
                int32_t dOff = dstBase / 2 + e;
                for (int32_t m = 0; m < innerMult; m++) {
                    outU64.SetValue(dOff, val);
                    dOff += hD;
                }
            }
        } else if (innerDim <= 3) {
            T cv0 = inData.GetValue(srcBase);
            T cv1 = (innerDim > 1) ? inData.GetValue(srcBase + 1) : cv0;
            T cv2 = (innerDim > 2) ? inData.GetValue(srcBase + 2) : cv0;
            int32_t dOff = dstBase;
            for (int32_t m = 0; m < innerMult; m++) {
                outBuf.SetValue(dOff, cv0);
                if (innerDim > 1)
                    outBuf.SetValue(dOff + 1, cv1);
                if (innerDim > 2)
                    outBuf.SetValue(dOff + 2, cv2);
                dOff += innerDim;
            }
        } else if (innerDim <= 12) {
            T cc[12];
            for (int32_t e = 0; e < innerDim; e++)
                cc[e] = inData.GetValue(srcBase + e);
            int32_t d = dstBase;
            for (int32_t m = 0; m < innerMult; m++) {
                switch (innerDim) {
                    case 5:
                        outBuf.SetValue(d, cc[0]);
                        outBuf.SetValue(d + 1, cc[1]);
                        outBuf.SetValue(d + 2, cc[2]);
                        outBuf.SetValue(d + 3, cc[3]);
                        outBuf.SetValue(d + 4, cc[4]);
                        break;
                    case 7:
                        outBuf.SetValue(d, cc[0]);
                        outBuf.SetValue(d + 1, cc[1]);
                        outBuf.SetValue(d + 2, cc[2]);
                        outBuf.SetValue(d + 3, cc[3]);
                        outBuf.SetValue(d + 4, cc[4]);
                        outBuf.SetValue(d + 5, cc[5]);
                        outBuf.SetValue(d + 6, cc[6]);
                        break;
                    case 9:
                        outBuf.SetValue(d, cc[0]);
                        outBuf.SetValue(d + 1, cc[1]);
                        outBuf.SetValue(d + 2, cc[2]);
                        outBuf.SetValue(d + 3, cc[3]);
                        outBuf.SetValue(d + 4, cc[4]);
                        outBuf.SetValue(d + 5, cc[5]);
                        outBuf.SetValue(d + 6, cc[6]);
                        outBuf.SetValue(d + 7, cc[7]);
                        outBuf.SetValue(d + 8, cc[8]);
                        break;
                    case 11:
                        outBuf.SetValue(d, cc[0]);
                        outBuf.SetValue(d + 1, cc[1]);
                        outBuf.SetValue(d + 2, cc[2]);
                        outBuf.SetValue(d + 3, cc[3]);
                        outBuf.SetValue(d + 4, cc[4]);
                        outBuf.SetValue(d + 5, cc[5]);
                        outBuf.SetValue(d + 6, cc[6]);
                        outBuf.SetValue(d + 7, cc[7]);
                        outBuf.SetValue(d + 8, cc[8]);
                        outBuf.SetValue(d + 9, cc[9]);
                        outBuf.SetValue(d + 10, cc[10]);
                        break;
                    default:
                        for (int32_t e = 0; e < innerDim; e++)
                            outBuf.SetValue(d + e, cc[e]);
                        break;
                }
                d += innerDim;
            }
        } else {
            for (int32_t e = 0; e < innerDim; e++) {
                T val = inData.GetValue(srcBase + e);
                int32_t dOff = dstBase + e;
                for (int32_t m = 0; m < innerMult; m++) {
                    outBuf.SetValue(dOff, val);
                    dOff += innerDim;
                }
            }
        }
    } else {
        bool packed = false;
        if constexpr (sizeof(T) == 2) {
            if ((innerDim % 2) == 0) {
                packed = true;
                auto inU32 = inData.template ReinterpretCast<uint32_t>();
                auto outU32 = outBuf.template ReinterpretCast<uint32_t>();
                int32_t hD = innerDim / 2;
                uint32_t pk[6];
                for (int32_t e = 0; e < hD; e++)
                    pk[e] = inU32.GetValue(srcBase / 2 + e);
                int32_t d = dstBase / 2;
                for (int32_t m = 0; m < innerMult; m++) {
                    switch (hD) {
                        case 1:
                            outU32.SetValue(d, pk[0]);
                            break;
                        case 2:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            break;
                        case 3:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            outU32.SetValue(d + 2, pk[2]);
                            break;
                        case 4:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            outU32.SetValue(d + 2, pk[2]);
                            outU32.SetValue(d + 3, pk[3]);
                            break;
                        case 5:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            outU32.SetValue(d + 2, pk[2]);
                            outU32.SetValue(d + 3, pk[3]);
                            outU32.SetValue(d + 4, pk[4]);
                            break;
                        case 6:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            outU32.SetValue(d + 2, pk[2]);
                            outU32.SetValue(d + 3, pk[3]);
                            outU32.SetValue(d + 4, pk[4]);
                            outU32.SetValue(d + 5, pk[5]);
                            break;
                        default:
                            for (int32_t e = 0; e < hD; e++)
                                outU32.SetValue(d + e, pk[e]);
                            break;
                    }
                    d += hD;
                }
            }
        } else if constexpr (sizeof(T) == 1) {
            if ((innerDim % 4) == 0 && innerDim >= 4) {
                packed = true;
                auto inU32 = inData.template ReinterpretCast<uint32_t>();
                auto outU32 = outBuf.template ReinterpretCast<uint32_t>();
                int32_t qD = innerDim / 4;
                uint32_t pk[3];
                for (int32_t e = 0; e < qD; e++)
                    pk[e] = inU32.GetValue(srcBase / 4 + e);
                int32_t d = dstBase / 4;
                for (int32_t m = 0; m < innerMult; m++) {
                    switch (qD) {
                        case 1:
                            outU32.SetValue(d, pk[0]);
                            break;
                        case 2:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            break;
                        case 3:
                            outU32.SetValue(d, pk[0]);
                            outU32.SetValue(d + 1, pk[1]);
                            outU32.SetValue(d + 2, pk[2]);
                            break;
                        default:
                            for (int32_t e = 0; e < qD; e++)
                                outU32.SetValue(d + e, pk[e]);
                            break;
                    }
                    d += qD;
                }
            } else if ((innerDim % 2) == 0) {
                packed = true;
                auto inU16 = inData.template ReinterpretCast<uint16_t>();
                auto outU16 = outBuf.template ReinterpretCast<uint16_t>();
                int32_t hD = innerDim / 2;
                uint16_t pk[6];
                for (int32_t e = 0; e < hD; e++)
                    pk[e] = inU16.GetValue(srcBase / 2 + e);
                int32_t d = dstBase / 2;
                for (int32_t m = 0; m < innerMult; m++) {
                    switch (hD) {
                        case 1:
                            outU16.SetValue(d, pk[0]);
                            break;
                        case 2:
                            outU16.SetValue(d, pk[0]);
                            outU16.SetValue(d + 1, pk[1]);
                            break;
                        case 3:
                            outU16.SetValue(d, pk[0]);
                            outU16.SetValue(d + 1, pk[1]);
                            outU16.SetValue(d + 2, pk[2]);
                            break;
                        case 4:
                            outU16.SetValue(d, pk[0]);
                            outU16.SetValue(d + 1, pk[1]);
                            outU16.SetValue(d + 2, pk[2]);
                            outU16.SetValue(d + 3, pk[3]);
                            break;
                        case 5:
                            outU16.SetValue(d, pk[0]);
                            outU16.SetValue(d + 1, pk[1]);
                            outU16.SetValue(d + 2, pk[2]);
                            outU16.SetValue(d + 3, pk[3]);
                            outU16.SetValue(d + 4, pk[4]);
                            break;
                        case 6:
                            outU16.SetValue(d, pk[0]);
                            outU16.SetValue(d + 1, pk[1]);
                            outU16.SetValue(d + 2, pk[2]);
                            outU16.SetValue(d + 3, pk[3]);
                            outU16.SetValue(d + 4, pk[4]);
                            outU16.SetValue(d + 5, pk[5]);
                            break;
                        default:
                            for (int32_t e = 0; e < hD; e++)
                                outU16.SetValue(d + e, pk[e]);
                            break;
                    }
                    d += hD;
                }
            }
        }
        if (!packed) {
            if (innerDim <= 3) {
                T cv0 = inData.GetValue(srcBase);
                T cv1 = (innerDim > 1) ? inData.GetValue(srcBase + 1) : cv0;
                T cv2 = (innerDim > 2) ? inData.GetValue(srcBase + 2) : cv0;
                int32_t dOff = dstBase;
                for (int32_t m = 0; m < innerMult; m++) {
                    outBuf.SetValue(dOff, cv0);
                    if (innerDim > 1)
                        outBuf.SetValue(dOff + 1, cv1);
                    if (innerDim > 2)
                        outBuf.SetValue(dOff + 2, cv2);
                    dOff += innerDim;
                }
            } else if (innerDim <= 12) {
                T cc[12];
                for (int32_t e = 0; e < innerDim; e++)
                    cc[e] = inData.GetValue(srcBase + e);
                int32_t d = dstBase;
                for (int32_t m = 0; m < innerMult; m++) {
                    switch (innerDim) {
                        case 4:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            break;
                        case 5:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            break;
                        case 6:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            break;
                        case 7:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            outBuf.SetValue(d + 6, cc[6]);
                            break;
                        case 8:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            outBuf.SetValue(d + 6, cc[6]);
                            outBuf.SetValue(d + 7, cc[7]);
                            break;
                        case 9:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            outBuf.SetValue(d + 6, cc[6]);
                            outBuf.SetValue(d + 7, cc[7]);
                            outBuf.SetValue(d + 8, cc[8]);
                            break;
                        case 10:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            outBuf.SetValue(d + 6, cc[6]);
                            outBuf.SetValue(d + 7, cc[7]);
                            outBuf.SetValue(d + 8, cc[8]);
                            outBuf.SetValue(d + 9, cc[9]);
                            break;
                        case 11:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            outBuf.SetValue(d + 6, cc[6]);
                            outBuf.SetValue(d + 7, cc[7]);
                            outBuf.SetValue(d + 8, cc[8]);
                            outBuf.SetValue(d + 9, cc[9]);
                            outBuf.SetValue(d + 10, cc[10]);
                            break;
                        case 12:
                            outBuf.SetValue(d, cc[0]);
                            outBuf.SetValue(d + 1, cc[1]);
                            outBuf.SetValue(d + 2, cc[2]);
                            outBuf.SetValue(d + 3, cc[3]);
                            outBuf.SetValue(d + 4, cc[4]);
                            outBuf.SetValue(d + 5, cc[5]);
                            outBuf.SetValue(d + 6, cc[6]);
                            outBuf.SetValue(d + 7, cc[7]);
                            outBuf.SetValue(d + 8, cc[8]);
                            outBuf.SetValue(d + 9, cc[9]);
                            outBuf.SetValue(d + 10, cc[10]);
                            outBuf.SetValue(d + 11, cc[11]);
                            break;
                        default:
                            for (int32_t e = 0; e < innerDim; e++)
                                outBuf.SetValue(d + e, cc[e]);
                            break;
                    }
                    d += innerDim;
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    T srcBuf[64];
                    int32_t readLen = (innerDim <= 64) ? innerDim : 64;
                    for (int32_t e = 0; e < readLen; e++)
                        srcBuf[e] = inData.GetValue(srcBase + e);
                    int32_t oiDimLocal = innerDim * innerMult;
                    auto outU32 = outBuf.template ReinterpretCast<uint32_t>();
                    int32_t leading = (4 - (dstBase % 4)) % 4;
                    if (leading > oiDimLocal)
                        leading = oiDimLocal;
                    for (int32_t tt = 0; tt < leading; tt++) {
                        outBuf.SetValue(dstBase + tt, srcBuf[tt % readLen]);
                    }
                    int32_t alignedOff = dstBase + leading;
                    int32_t remaining = oiDimLocal - leading;
                    int32_t nU32 = remaining / 4;
                    for (int32_t k = 0; k < nU32; k++) {
                        int32_t p = leading + 4 * k;
                        uint32_t val = static_cast<uint8_t>(srcBuf[p % readLen]) |
                                       (static_cast<uint32_t>(static_cast<uint8_t>(srcBuf[(p + 1) % readLen])) << 8) |
                                       (static_cast<uint32_t>(static_cast<uint8_t>(srcBuf[(p + 2) % readLen])) << 16) |
                                       (static_cast<uint32_t>(static_cast<uint8_t>(srcBuf[(p + 3) % readLen])) << 24);
                        outU32.SetValue(alignedOff / 4 + k, val);
                    }
                    int32_t tailOff = leading + nU32 * 4;
                    for (int32_t tt = tailOff; tt < oiDimLocal; tt++) {
                        outBuf.SetValue(dstBase + tt, srcBuf[tt % readLen]);
                    }
                } else {
                    for (int32_t e = 0; e < innerDim; e++) {
                        T val = inData.GetValue(srcBase + e);
                        int32_t dOff = dstBase + e;
                        for (int32_t m = 0; m < innerMult; m++) {
                            outBuf.SetValue(dOff, val);
                            dOff += innerDim;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline int32_t GatherBuildRows(
    LocalTensor<T>& outBuf, const LocalTensor<T>& inData, int32_t numRows, int32_t innerDim, int32_t innerMult,
    int32_t srcStride, int32_t outputInnerDim, int32_t alignElems)
{
    if constexpr (sizeof(T) < 2) {
        return -1;
    }
    int32_t oiDim = outputInnerDim;
    int32_t rowBytes = oiDim * static_cast<int32_t>(sizeof(T));
    int32_t gcdVal = rowBytes;
    int32_t bv = 32;
    while (bv) {
        int32_t t = bv;
        bv = gcdVal % bv;
        gcdVal = t;
    }
    int32_t superRowCount = 32 / gcdVal;
    if (superRowCount > numRows)
        superRowCount = numRows;
    if (superRowCount < 2)
        superRowCount = 1;
    int32_t superRowElems = superRowCount * oiDim;

    int32_t offsetU32 = ((superRowElems + 7) / 8) * 8;
    int32_t offElemsT = (offsetU32 * 4 + static_cast<int32_t>(sizeof(T)) - 1) / static_cast<int32_t>(sizeof(T));
    int32_t gatherStart = ((offElemsT + alignElems - 1) / alignElems) * alignElems;

    auto gOff = outBuf[0].template ReinterpretCast<uint32_t>();
    for (int32_t k = 0; k < superRowElems; k++) {
        int32_t rowInBatch = k / oiDim;
        int32_t srcElem = (k % oiDim) % innerDim;
        gOff.SetValue(
            k,
            static_cast<uint32_t>(
                rowInBatch * srcStride * static_cast<int32_t>(sizeof(T)) + srcElem * static_cast<int32_t>(sizeof(T))));
    }
    PipeBarrier<PIPE_V>();

    int32_t fullBatches = numRows / superRowCount;
    for (int32_t b = 0; b < fullBatches; b++) {
        int32_t batchSrc = b * superRowCount * srcStride;
        int32_t batchDst = gatherStart + b * superRowElems;
        Gather(
            outBuf[batchDst], inData[batchSrc], gOff, static_cast<uint32_t>(0), static_cast<uint32_t>(superRowElems));
    }
    PipeBarrier<PIPE_V>();
    ResetMask();

    int32_t tailStart = fullBatches * superRowCount;
    for (int32_t t = tailStart; t < numRows; t++) {
        BuildInnerRow(outBuf, inData, t * srcStride, gatherStart + t * oiDim, innerDim, innerMult);
    }
    return gatherStart;
}

template <typename T>
__aicore__ inline int32_t GatherBuildRowsU8(
    LocalTensor<T>& outBuf, const LocalTensor<T>& inData, int32_t numRows, int32_t innerDim, int32_t innerMult,
    int32_t srcStride, int32_t outputInnerDim, int32_t alignElems)
{
    static_assert(sizeof(T) == 1, "GatherBuildRowsU8 requires sizeof(T)==1");
    if (innerDim % 2 != 0 || outputInnerDim % 2 != 0) {
        return -1;
    }

    int32_t oiDim_u16 = outputInnerDim / 2;
    int32_t srcStride_u16 = srcStride / 2;
    constexpr int32_t ALIGN_U16 = DATA_BLOCK_BYTES / 2;

    int32_t offsetU32 = ((oiDim_u16 + 7) / 8) * 8;
    int32_t offBytes = offsetU32 * 4;
    int32_t offElemsU16 = (offBytes + 1) / 2;
    int32_t gatherStart_u16 = ((offElemsU16 + ALIGN_U16 - 1) / ALIGN_U16) * ALIGN_U16;
    int32_t gatherStart_u8 = gatherStart_u16 * 2;

    auto outU16 = outBuf.template ReinterpretCast<uint16_t>();
    auto inU16 = inData.template ReinterpretCast<uint16_t>();
    auto gOff = outBuf[0].template ReinterpretCast<uint32_t>();

    for (int32_t k = 0; k < oiDim_u16; k++) {
        gOff.SetValue(k, static_cast<uint32_t>((2 * k) % innerDim));
    }
    PipeBarrier<PIPE_V>();

    for (int32_t r = 0; r < numRows; r++) {
        Gather(
            outU16[gatherStart_u16 + r * oiDim_u16], inU16[r * srcStride_u16], gOff, static_cast<uint32_t>(0),
            static_cast<uint32_t>(oiDim_u16));
    }
    PipeBarrier<PIPE_V>();
    ResetMask();

    return gatherStart_u8;
}

template <typename T>
class TileOpImpl {
public:
    __aicore__ inline TileOpImpl() {}

    template <uint32_t schMode = 0>
    __aicore__ inline void Init(GM_ADDR inputGm, GM_ADDR outputGm, const TileTilingData* tilingPtr, TPipe* pipeIn);
    template <uint32_t schMode = 0>
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSplitMult(
        int32_t innerDimAligned, int32_t bufElems, bool isInnerAligned, int32_t inputBytes);

    __aicore__ inline void ProcessBuild(
        int32_t maxBatch, int32_t innerDimAligned, int32_t outputDimAligned, bool isOutputAligned);
    __aicore__ inline void ProcessPerRow(
        int32_t maxBatch, int32_t innerDimAligned, bool isInnerAligned, int32_t inputBytes);
    __aicore__ inline void ProcessLargeInner(int32_t bufElems);
    __aicore__ inline void ProcessScalarBuildFast(int32_t innerDimAligned, int32_t bufElems);
    __aicore__ inline void ProcessBuildOnceAmplify(int32_t innerDimAligned, int32_t bufElems);
    __aicore__ inline void ProcessDmaTemplateBuild(int32_t innerDimAligned, int32_t bufElems);
    __aicore__ inline void ProcessRowReadbackAmplify(int32_t innerDimAligned, int32_t bufElems);
    __aicore__ inline void ProcessVecGatherBuild(int32_t innerDimAligned, int32_t bufElems);
    __aicore__ inline int32_t SrcOff(int32_t outerIdx);
    __aicore__ inline void ProcessFlatResident();

    __aicore__ inline void WaitMTE3();

    GlobalTensor<T> gmIn, gmOut;
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    int32_t numDims, inputShape[MAX_DIM], multiples[MAX_DIM], outputShape[MAX_DIM];
    int32_t inputStrides[MAX_DIM], outputStrides[MAX_DIM];
    int32_t totalOutputElems, totalInputElems, elemBytes, alignElems;
    int32_t innerDim, innerMult, outputInnerDim, outerCount;
    int32_t myStartRow, myEndRow, myStartMult, myEndMult;
    bool splitByMult;
    int32_t blockDimVal;
    int32_t bufferBytes;
    int32_t ubTotalBytes;
    int32_t tilingRepeatPeriod, tilingRepeatInputPeriod;
    int32_t tilingPeriodsPerSource;
    int32_t tilingNUniqueSources;
};

template <typename T>
template <uint32_t schMode>
__aicore__ inline void TileOpImpl<T>::Init(
    GM_ADDR inputGm, GM_ADDR outputGm, const TileTilingData* tilingPtr, TPipe* pipeIn)
{
    pipe = pipeIn;
    numDims = tilingPtr->numDims;
    totalOutputElems = tilingPtr->totalOutputElems;
    totalInputElems = tilingPtr->totalInputElems;
    elemBytes = tilingPtr->elemBytes;
    if (elemBytes <= 0) {
        elemBytes = 1;
    }
    blockDimVal = tilingPtr->blockDim;
    bufferBytes = tilingPtr->ubSize > 0 ? tilingPtr->ubSize : DEFAULT_BUFFER_BYTES;
    tilingRepeatPeriod = tilingPtr->repeatPeriod;
    tilingRepeatInputPeriod = tilingPtr->repeatInputPeriod;
    tilingPeriodsPerSource = tilingPtr->periodsPerSource;
    tilingNUniqueSources = tilingPtr->nUniqueSources;
    for (int32_t idx = 0; idx < numDims; idx++) {
        inputShape[idx] = tilingPtr->inputShape[idx];
        multiples[idx] = tilingPtr->multiples[idx];
        outputShape[idx] = tilingPtr->outputShape[idx];
        inputStrides[idx] = tilingPtr->inputStrides[idx];
        outputStrides[idx] = tilingPtr->outputStrides[idx];
    }
    innerDim = inputShape[numDims - 1];
    innerMult = multiples[numDims - 1];
    if (innerDim <= 0) {
        innerDim = 1;
    }
    if (innerMult <= 0) {
        innerMult = 1;
    }
    outputInnerDim = innerDim * innerMult;
    outerCount = totalOutputElems / outputInnerDim;
    constexpr int32_t ALIGN = DATA_BLOCK_BYTES / static_cast<int32_t>(sizeof(T));
    alignElems = ALIGN;

    int32_t coreId = GetBlockIdx();
    int32_t totalWork = outerCount * innerMult;
    int32_t innerDimAlignedInit = ((innerDim + alignElems - 1) / alignElems) * alignElems;
    int32_t bufElemsInit = bufferBytes / elemBytes;
    splitByMult = (outerCount < blockDimVal && innerMult > 1 && innerDimAlignedInit <= bufElemsInit);

    if (splitByMult) {
        int32_t workPerCore = (totalWork + blockDimVal - 1) / blockDimVal;
        int32_t startWork = coreId * workPerCore;
        int32_t endWork = startWork + workPerCore;
        if (endWork > totalWork) {
            endWork = totalWork;
        }
        if (startWork >= totalWork) {
            myStartRow = 0;
            myEndRow = 0;
            myStartMult = 0;
            myEndMult = 0;
        } else {
            myStartRow = startWork / innerMult;
            myStartMult = startWork % innerMult;
            myEndRow = (endWork - 1) / innerMult;
            myEndMult = (endWork - 1) % innerMult + 1;
            if (myStartRow == myEndRow) {
                myEndRow = myStartRow + 1;
            } else {
                myEndRow = myEndRow + 1;
            }
        }
    } else {
        int32_t rowsPerCore = (outerCount + blockDimVal - 1) / blockDimVal;
        int32_t rowBytes = outputInnerDim * elemBytes;
        if (rowBytes > 0 && rowBytes % DATA_BLOCK_BYTES != 0) {
            int32_t g = rowBytes;
            int32_t bv = DATA_BLOCK_BYTES;
            while (bv) {
                int32_t t = bv;
                bv = g % bv;
                g = t;
            }
            int32_t alignStep = DATA_BLOCK_BYTES / g;
            if (alignStep > 1 && rowsPerCore > alignStep) {
                rowsPerCore = ((rowsPerCore + alignStep - 1) / alignStep) * alignStep;
            }
        }
        myStartRow = coreId * rowsPerCore;
        myEndRow = myStartRow + rowsPerCore;
        if (myEndRow > outerCount) {
            myEndRow = outerCount;
        }
        if (myStartRow >= outerCount) {
            myStartRow = 0;
            myEndRow = 0;
        }
        myStartMult = 0;
        myEndMult = innerMult;
    }

    gmIn.SetGlobalBuffer((__gm__ T*)inputGm);
    gmOut.SetGlobalBuffer((__gm__ T*)outputGm);
    ubTotalBytes = 2 * bufferBytes;
    bool extendInQue = false;
    if constexpr (schMode == TILE_TPL_SCH_MODE_READBACK) {
        constexpr int32_t ALIGN_CHK = DATA_BLOCK_BYTES / static_cast<int32_t>(sizeof(T));
        int32_t idAligned = ((innerDim + ALIGN_CHK - 1) / ALIGN_CHK) * ALIGN_CHK;
        int32_t halfBufElems = bufferBytes / elemBytes;
        bool dmaBuildCond =
            (innerMult > 1 && idAligned * innerMult <= halfBufElems && innerDim * elemBytes >= DATA_BLOCK_BYTES);
        bool gatherHybridOk = false;
        if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
            gatherHybridOk = true;
        }
        extendInQue = dmaBuildCond && !gatherHybridOk;
    }
    if (extendInQue) {
        pipe->InitBuffer(inQue, 1, ubTotalBytes);
    } else {
        pipe->InitBuffer(inQue, 1, bufferBytes);
        pipe->InitBuffer(outQue, 1, bufferBytes);
    }
}

template <typename T>
template <uint32_t schMode>
__aicore__ inline void TileOpImpl<T>::Process()
{
    int32_t innerDimAligned = ((innerDim + alignElems - 1) / alignElems) * alignElems;
    int32_t bufElems = bufferBytes / elemBytes;

    if constexpr (schMode == 0) {
        if (innerDimAligned > bufElems) {
            ProcessLargeInner(bufElems);
            return;
        }
    }

    if (myStartRow >= myEndRow) {
        return;
    }

    if constexpr (schMode == 2) {
        ProcessBuildOnceAmplify(innerDimAligned, bufElems);
        return;
    }

    if constexpr (schMode == 3) {
        if (innerDim == 1 && innerMult > alignElems) {
            int32_t srcOff = SrcOff(myStartRow);
            int32_t totalFill = (myEndRow - myStartRow) * outputInnerDim;
            if (totalFill <= 0) {
                return;
            }
            LocalTensor<T> inBuf = inQue.AllocTensor<T>();
            CopyIn(inBuf, gmIn[srcOff], alignElems);
            inQue.EnQue(inBuf);
            LocalTensor<T> inData = inQue.DeQue<T>();
            T val = inData.GetValue(0);
            inQue.FreeTensor(inData);
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();
            if constexpr (sizeof(T) >= sizeof(uint16_t) && sizeof(T) <= sizeof(uint32_t)) {
                Duplicate(outBuf, val, totalFill);
            } else {
                for (int32_t e = 0; e < totalFill && e < bufElems; e++) {
                    outBuf.SetValue(e, val);
                }
            }
            outQue.EnQue(outBuf);
            LocalTensor<T> outData = outQue.DeQue<T>();
            int32_t gmBase = myStartRow * outputInnerDim;
            int32_t written = 0;
            while (written < totalFill) {
                int32_t chunk = bufElems;
                if (written + chunk > totalFill) {
                    chunk = totalFill - written;
                }
                CopyOut(gmOut[gmBase + written], outData[written], chunk);
                written += chunk;
            }
            WaitMTE3();
            outQue.FreeTensor(outData);
            return;
        }
        ProcessDmaTemplateBuild(innerDimAligned, bufElems);
        return;
    }

    if constexpr (schMode == 4) {
        int32_t rbBufElems = bufElems;
        constexpr int32_t ALIGN_CHK = DATA_BLOCK_BYTES / static_cast<int32_t>(sizeof(T));
        int32_t idAligned = ((innerDim + ALIGN_CHK - 1) / ALIGN_CHK) * ALIGN_CHK;
        bool dmaBuildCheck =
            (innerMult > 1 && idAligned * innerMult <= bufElems && innerDim * elemBytes >= DATA_BLOCK_BYTES);
        bool gatherHybridCheck = false;
        if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
            gatherHybridCheck = (dmaBuildCheck && innerDim != idAligned);
        }
        if (dmaBuildCheck && !gatherHybridCheck) {
            rbBufElems = ubTotalBytes / elemBytes;
        }
        ProcessRowReadbackAmplify(innerDimAligned, rbBufElems);
        return;
    }

    if constexpr (schMode == 5) {
        ProcessVecGatherBuild(innerDimAligned, bufElems);
        return;
    }

    int32_t outputDimAligned = ((outputInnerDim + alignElems - 1) / alignElems) * alignElems;
    int32_t inputBytes = innerDim * elemBytes;
    bool isInnerAligned = (innerDim == innerDimAligned);
    bool isOutputAligned = (outputInnerDim == outputDimAligned);

    int32_t inputAligned = ((totalInputElems + alignElems - 1) / alignElems) * alignElems;
    int32_t seedRows = totalInputElems / innerDim;
    int32_t inputUbElems = inputAligned;
    int32_t cacheUbElems = seedRows * outputDimAligned;
    int32_t rowsPerCore = (outerCount + blockDimVal - 1) / blockDimVal;
    bool canCacheFR = !splitByMult && !isInnerAligned && innerMult > 1 && inputUbElems <= bufElems &&
                      cacheUbElems <= bufElems && rowsPerCore >= seedRows * innerDim;
    if (canCacheFR) {
        ProcessFlatResident();
        return;
    }
    if (!splitByMult && inputAligned <= bufElems && outputDimAligned <= bufElems &&
        totalOutputElems >= totalInputElems * 16 && totalInputElems <= 256) {
        ProcessFlatResident();
        return;
    }

    if (splitByMult) {
        ProcessSplitMult(innerDimAligned, bufElems, isInnerAligned, inputBytes);
    } else if (!isInnerAligned && innerMult > 1 && innerDim <= 12 && outputInnerDim <= bufElems) {
        ProcessScalarBuildFast(innerDimAligned, bufElems);
    } else {
        bool canBuild = (outputDimAligned <= bufElems) && isInnerAligned && (innerMult > 2);
        int32_t maxBatch = canBuild ? (bufElems / outputDimAligned) : (bufElems / innerDimAligned);
        if (maxBatch < 1) {
            maxBatch = 1;
        }
        if (canBuild) {
            ProcessBuild(maxBatch, innerDimAligned, outputDimAligned, isOutputAligned);
        } else {
            ProcessPerRow(maxBatch, innerDimAligned, isInnerAligned, inputBytes);
        }
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessSplitMult(
    int32_t innerDimAligned, int32_t bufElems, bool isInnerAligned, int32_t inputBytes)
{
    for (int32_t outer = myStartRow; outer < myEndRow; outer++) {
        int32_t srcOff = SrcOff(outer);
        int32_t dstBase = outer * outputInnerDim;
        int32_t multStart = (outer == myStartRow) ? myStartMult : 0;
        int32_t multEnd = (outer == myEndRow - 1) ? myEndMult : innerMult;
        int32_t multCount = multEnd - multStart;
        int32_t totalElems = multCount * innerDim;
        bool canDouble = (innerDim % alignElems == 0);
        LocalTensor<T> inBuf = inQue.AllocTensor<T>();
        CopyIn(inBuf, gmIn[srcOff], innerDimAligned);
        inQue.EnQue(inBuf);
        LocalTensor<T> inData = inQue.DeQue<T>();
        if (!canDouble && innerDim <= 12 && totalElems <= bufElems) {
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();
            if (innerDim <= 3) {
                T sv0 = inData.GetValue(0);
                T sv1 = (innerDim > 1) ? inData.GetValue(1) : sv0;
                T sv2 = (innerDim > 2) ? inData.GetValue(2) : sv0;
                for (int32_t m = 0; m < multCount; m++) {
                    int32_t base = m * innerDim;
                    outBuf.SetValue(base, sv0);
                    if (innerDim > 1) {
                        outBuf.SetValue(base + 1, sv1);
                    }
                    if (innerDim > 2) {
                        outBuf.SetValue(base + 2, sv2);
                    }
                }
            } else {
                T sv[12];
                for (int32_t e = 0; e < innerDim; e++) {
                    sv[e] = inData.GetValue(e);
                }
                for (int32_t m = 0; m < multCount; m++) {
                    int32_t base = m * innerDim;
                    for (int32_t e = 0; e < innerDim; e++) {
                        outBuf.SetValue(base + e, sv[e]);
                    }
                }
            }
            inQue.FreeTensor(inData);
            outQue.EnQue(outBuf);
            LocalTensor<T> outData = outQue.DeQue<T>();
            int32_t gmOffset = dstBase + multStart * innerDim;
            CopyOut(gmOut[gmOffset], outData, totalElems);
            WaitMTE3();
            outQue.FreeTensor(outData);
        } else {
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();
            DataCopy(outBuf, inData, innerDimAligned);
            inQue.FreeTensor(inData);
            int32_t curLen = innerDim;
            while (canDouble && curLen * 2 <= bufElems && curLen < totalElems) {
                int32_t copyAligned = ((curLen + alignElems - 1) / alignElems) * alignElems;
                DataCopy(outBuf[curLen], outBuf, copyAligned);
                curLen *= 2;
            }
            outQue.EnQue(outBuf);
            LocalTensor<T> outData = outQue.DeQue<T>();
            int32_t written = 0;
            while (written < totalElems) {
                int32_t chunk = curLen;
                if (written + chunk > totalElems) {
                    chunk = totalElems - written;
                }
                int32_t gmOffset = dstBase + multStart * innerDim + written;
                CopyOut(gmOut[gmOffset], outData, chunk);
                written += chunk;
            }
            WaitMTE3();
            outQue.FreeTensor(outData);
        }
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessBuild(
    int32_t maxBatch, int32_t innerDimAligned, int32_t outputDimAligned, bool isOutputAligned)
{
    int32_t outer = myStartRow;
    bool canBatchRead = (innerDim == innerDimAligned) && (myEndRow - myStartRow > 1) &&
                        (SrcOff(myStartRow + 1) - SrcOff(myStartRow) == innerDim);
    while (outer < myEndRow) {
        int32_t batchSize = myEndRow - outer;
        if (batchSize > maxBatch) {
            batchSize = maxBatch;
        }
        LocalTensor<T> inBuf = inQue.AllocTensor<T>();
        if (canBatchRead && batchSize > 1) {
            CopyIn(inBuf, gmIn[SrcOff(outer)], batchSize * innerDimAligned);
        } else {
            for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                CopyIn(inBuf[bIdx * innerDimAligned], gmIn[SrcOff(outer + bIdx)], innerDimAligned);
            }
        }
        inQue.EnQue(inBuf);
        LocalTensor<T> inData = inQue.DeQue<T>();
        LocalTensor<T> outBuf = outQue.AllocTensor<T>();
        for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            for (int32_t mIdx = 0; mIdx < innerMult; mIdx++) {
                DataCopy(
                    outBuf[bIdx * outputDimAligned + mIdx * innerDim], inData[bIdx * innerDimAligned], innerDimAligned);
            }
        }
        inQue.FreeTensor(inData);
        outQue.EnQue(outBuf);
        LocalTensor<T> outData = outQue.DeQue<T>();
        if (isOutputAligned && outputDimAligned == outputInnerDim) {
            CopyOut(gmOut[outer * outputInnerDim], outData, batchSize * outputInnerDim);
        } else {
            for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                int32_t dst = (outer + bIdx) * outputInnerDim;
                CopyOut(gmOut[dst], outData[bIdx * outputDimAligned], outputInnerDim);
            }
        }
        WaitMTE3();
        outQue.FreeTensor(outData);
        outer += batchSize;
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessPerRow(
    int32_t maxBatch, int32_t innerDimAligned, bool isInnerAligned, int32_t inputBytes)
{
    int32_t repeatPeriod = tilingRepeatPeriod;
    int32_t repeatInputPeriod = tilingRepeatInputPeriod;
    bool hasRepeat = (repeatPeriod > repeatInputPeriod && repeatInputPeriod > 0);
    int32_t numAmplify = hasRepeat ? (repeatPeriod / repeatInputPeriod) : 1;
    bool useAmplify = hasRepeat && (numAmplify * innerMult <= 100);

    int32_t outer = myStartRow;
    while (outer < myEndRow) {
        int32_t posInBlock = hasRepeat ? (outer % repeatPeriod) : 0;
        int32_t blockBase = hasRepeat ? (outer - posInBlock) : outer;

        bool isAmplified = useAmplify && (posInBlock >= repeatInputPeriod) && (blockBase >= myStartRow);
        if (isAmplified) {
            int32_t blockEnd = blockBase + repeatPeriod;
            if (blockEnd > myEndRow) {
                blockEnd = myEndRow;
            }
            outer = blockEnd;
            continue;
        }

        bool isRepeat = hasRepeat && (posInBlock >= repeatInputPeriod) && outer >= myStartRow + repeatInputPeriod;

        int32_t batchSize = myEndRow - outer;
        if (batchSize > maxBatch) {
            batchSize = maxBatch;
        }

        if (isRepeat) {
            int32_t srcRow = outer - repeatInputPeriod;
            int32_t repeatSrcGm = srcRow * outputInnerDim;
            if (batchSize > repeatInputPeriod) {
                batchSize = repeatInputPeriod;
            }
            int32_t remainInPeriod = repeatPeriod - posInBlock;
            if (batchSize > remainInPeriod) {
                batchSize = remainInPeriod;
            }
            int32_t dstGm = outer * outputInnerDim;
            int32_t copyElems = batchSize * outputInnerDim;
            int32_t bufElemsLocal = bufferBytes / elemBytes;
            LocalTensor<T> cpBuf = inQue.AllocTensor<T>();
            int32_t copied = 0;
            while (copied < copyElems) {
                int32_t chunk = copyElems - copied;
                if (chunk > bufElemsLocal) {
                    chunk = (bufElemsLocal / alignElems) * alignElems;
                }
                CopyIn(cpBuf, gmOut[repeatSrcGm + copied], chunk);
                event_t rdId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(rdId);
                WaitFlag<HardEvent::MTE2_MTE3>(rdId);
                CopyOut(gmOut[dstGm + copied], cpBuf, chunk);
                event_t wrId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(wrId);
                WaitFlag<HardEvent::MTE3_MTE2>(wrId);
                copied += chunk;
            }
            inQue.FreeTensor(cpBuf);
            outer += batchSize;
            continue;
        }

        if (useAmplify) {
            int32_t seedEnd = blockBase + repeatInputPeriod;
            if (seedEnd > outer && outer + batchSize > seedEnd) {
                batchSize = seedEnd - outer;
            } else if (seedEnd <= outer) {
                int32_t periodEndA = blockBase + repeatPeriod;
                if (outer + batchSize > periodEndA) {
                    batchSize = periodEndA - outer;
                }
            }
        }
        if (batchSize < 1) {
            batchSize = 1;
        }

        LocalTensor<T> inBuf = inQue.AllocTensor<T>();
        bool canBulkRead = isInnerAligned && (batchSize > 1) && (SrcOff(outer + 1) - SrcOff(outer) == innerDim) &&
                           (SrcOff(outer + batchSize - 1) == SrcOff(outer) + (batchSize - 1) * innerDim);
        if (canBulkRead) {
            CopyIn(inBuf, gmIn[SrcOff(outer)], batchSize * innerDimAligned);
        } else {
            for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                CopyIn(inBuf[bIdx * innerDimAligned], gmIn[SrcOff(outer + bIdx)], innerDimAligned);
            }
        }
        inQue.EnQue(inBuf);
        LocalTensor<T> inData = inQue.DeQue<T>();

        LocalTensor<T> outBuf = outQue.AllocTensor<T>();
        for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            DataCopy(outBuf[bIdx * innerDimAligned], inData[bIdx * innerDimAligned], innerDimAligned);
        }
        inQue.FreeTensor(inData);
        outQue.EnQue(outBuf);
        LocalTensor<T> outData = outQue.DeQue<T>();

        int32_t amplifyCount = useAmplify ? numAmplify : 1;
        int32_t periodEnd = useAmplify ? (blockBase + repeatPeriod) : (myEndRow + 1);
        for (int32_t p = 0; p < amplifyCount; p++) {
            int32_t destRow = useAmplify ? (blockBase + p * repeatInputPeriod + posInBlock) : outer;
            if (destRow >= periodEnd) {
                break;
            }
            if (destRow >= myEndRow) {
                break;
            }
            int32_t writeRows = batchSize;
            if (destRow + writeRows > myEndRow) {
                writeRows = myEndRow - destRow;
            }
            if (destRow + writeRows > periodEnd) {
                writeRows = periodEnd - destRow;
            }
            if (writeRows <= 0) {
                break;
            }

            for (int32_t bIdx = 0; bIdx < writeRows; bIdx++) {
                int32_t dst = (destRow + bIdx) * outputInnerDim;
                for (int32_t mIdx = 0; mIdx < innerMult; mIdx++) {
                    CopyOut(gmOut[dst + mIdx * innerDim], outData[bIdx * innerDimAligned], innerDim);
                }
            }
        }

        WaitMTE3();
        outQue.FreeTensor(outData);
        outer += batchSize;
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessBuildOnceAmplify(int32_t innerDimAligned, int32_t bufElems)
{
    int32_t repeatPeriod = tilingRepeatPeriod;
    int32_t repeatInputPeriod = tilingRepeatInputPeriod;
    int32_t numAmplify = repeatPeriod / repeatInputPeriod;
    int32_t seedOutElems = repeatInputPeriod * outputInnerDim;
    int32_t seedInElems = (repeatInputPeriod - 1) * innerDim + innerDimAligned;

    int32_t totalPeriods = outerCount / repeatPeriod;
    int32_t remainderRows = outerCount - totalPeriods * repeatPeriod;
    int32_t periodsPS = tilingPeriodsPerSource;
    bool srcGrp = (periodsPS > 0 && totalPeriods >= blockDimVal);
    int32_t nSrc = srcGrp ? (totalPeriods / periodsPS) : 0;
    int32_t srcPerCore = 1;
    if (srcGrp) {
        int32_t coreId = GetBlockIdx();
        if (nSrc <= blockDimVal) {
            if (coreId >= nSrc) {
                if (remainderRows > 0 && coreId == nSrc) {
                    myStartRow = totalPeriods * repeatPeriod;
                    myEndRow = outerCount;
                    ProcessScalarBuildFast(innerDimAligned, bufElems);
                }
                return;
            }
        } else {
            srcPerCore = (nSrc + blockDimVal - 1) / blockDimVal;
            int32_t mySrcStart = coreId * srcPerCore;
            if (mySrcStart >= nSrc) {
                if (remainderRows > 0 && coreId == (nSrc + srcPerCore - 1) / srcPerCore) {
                    myStartRow = totalPeriods * repeatPeriod;
                    myEndRow = outerCount;
                    ProcessScalarBuildFast(innerDimAligned, bufElems);
                }
                return;
            }
        }
        myStartRow = 0;
        myEndRow = totalPeriods * repeatPeriod;
    } else if (totalPeriods > 1) {
        int32_t coreId = GetBlockIdx();
        int32_t periodsPerCore = (totalPeriods + blockDimVal - 1) / blockDimVal;
        int32_t myFirstPeriod = coreId * periodsPerCore;
        int32_t myLastPeriod = myFirstPeriod + periodsPerCore;
        if (myLastPeriod > totalPeriods)
            myLastPeriod = totalPeriods;
        if (myFirstPeriod >= totalPeriods) {
            if (remainderRows > 0 && coreId == totalPeriods) {
                myStartRow = totalPeriods * repeatPeriod;
                myEndRow = outerCount;
                ProcessScalarBuildFast(innerDimAligned, bufElems);
            }
            return;
        }
        myStartRow = myFirstPeriod * repeatPeriod;
        myEndRow = myLastPeriod * repeatPeriod;
        if (myLastPeriod == totalPeriods && remainderRows > 0) {
            myEndRow = outerCount;
        }
    }

    int32_t firstFullPeriod = ((myStartRow + repeatPeriod - 1) / repeatPeriod) * repeatPeriod;
    int32_t lastFullPeriodEnd = (myEndRow / repeatPeriod) * repeatPeriod;
    bool seedBulkOk =
        (repeatInputPeriod <= 1) || (SrcOff(1) - SrcOff(0) == innerDim &&
                                     SrcOff(repeatInputPeriod - 1) == SrcOff(0) + (repeatInputPeriod - 1) * innerDim);
    int32_t seedSrcStride = seedBulkOk ? innerDim : innerDimAligned;

    int32_t fullPeriodElems = repeatPeriod * outputInnerDim;
    int32_t fullPeriodAligned = ((fullPeriodElems + alignElems - 1) / alignElems) * alignElems;
    bool useFullPeriodCache =
        srcGrp && nSrc <= blockDimVal && numAmplify >= 3 && fullPeriodAligned <= bufElems && periodsPS > 1;

    if (useFullPeriodCache) {
        int32_t coreId = GetBlockIdx();
        for (int32_t lsrc = 0; lsrc < srcPerCore; lsrc++) {
            int32_t gsrc = coreId * srcPerCore + lsrc;
            if (gsrc >= nSrc)
                break;
            int32_t firstPb = gsrc * repeatPeriod;
            int32_t pSrc = SrcOff(firstPb);
            LocalTensor<T> inBuf = inQue.AllocTensor<T>();
            if (seedBulkOk) {
                int32_t ba = ((seedInElems + alignElems - 1) / alignElems) * alignElems;
                CopyIn(inBuf, gmIn[pSrc], ba);
            } else {
                for (int32_t s = 0; s < repeatInputPeriod; s++)
                    CopyIn(inBuf[s * innerDimAligned], gmIn[pSrc + SrcOff(s)], innerDimAligned);
            }
            inQue.EnQue(inBuf);
            LocalTensor<T> inData = inQue.DeQue<T>();
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();
            if (innerMult == 1) {
                int32_t ce = repeatInputPeriod * innerDim;
                int32_t ca = ((ce + alignElems - 1) / alignElems) * alignElems;
                if (seedBulkOk && ca <= bufElems)
                    DataCopy(outBuf, inData, ca);
                else
                    for (int32_t b = 0; b < repeatInputPeriod; b++)
                        DataCopy(outBuf[b * outputInnerDim], inData[b * seedSrcStride], innerDimAligned);
            } else {
                for (int32_t b = 0; b < repeatInputPeriod; b++)
                    BuildInnerRow(outBuf, inData, b * seedSrcStride, b * outputInnerDim, innerDim, innerMult);
            }
            inQue.FreeTensor(inData);
            outQue.EnQue(outBuf);
            LocalTensor<T> outData2 = outQue.DeQue<T>();
            for (int32_t p = 0; p < numAmplify; p++)
                CopyOut(gmOut[(firstPb + p * repeatInputPeriod) * outputInnerDim], outData2, seedOutElems);
            WaitMTE3();
            outQue.FreeTensor(outData2);
            LocalTensor<T> rpBuf = inQue.AllocTensor<T>();
            CopyIn(rpBuf, gmOut[firstPb * outputInnerDim], fullPeriodAligned);
            event_t rdEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(rdEv);
            WaitFlag<HardEvent::MTE2_MTE3>(rdEv);
            inQue.EnQue(rpBuf);
            LocalTensor<T> rpData = inQue.DeQue<T>();
            for (int32_t sp = 1; sp < periodsPS; sp++) {
                int32_t pb2 = (sp * nSrc + gsrc) * repeatPeriod;
                CopyOut(gmOut[pb2 * outputInnerDim], rpData, fullPeriodElems);
            }
            WaitMTE3();
            inQue.FreeTensor(rpData);
        }
        if (remainderRows > 0) {
            int32_t lastCore = (nSrc - 1) / srcPerCore;
            if (GetBlockIdx() == lastCore) {
                myStartRow = totalPeriods * repeatPeriod;
                myEndRow = outerCount;
                ProcessScalarBuildFast(innerDimAligned, bufElems);
            }
        }
        return;
    }

    int32_t extStart = (myStartRow / repeatPeriod) * repeatPeriod;
    int32_t extEnd = lastFullPeriodEnd;
    if (lastFullPeriodEnd < myEndRow) {
        extEnd = lastFullPeriodEnd + repeatPeriod;
    }
    int32_t scalarFallbackStart = myStartRow;
    int32_t scalarFallbackEnd = myStartRow;
    int32_t seedOutputOffset = 0;

    if (extStart < extEnd) {
        int32_t nUS = tilingNUniqueSources;
        int32_t linearN = (extEnd - extStart) / repeatPeriod;
        int32_t seedOutAligned = ((seedOutElems + alignElems - 1) / alignElems) * alignElems;
        bool useMultiTplCache = !srcGrp && nUS > 1 && nUS <= 64 && nUS < linearN &&
                                static_cast<int64_t>(nUS) * seedOutAligned <= bufElems && seedInElems <= bufElems;
        if (useMultiTplCache) {
            int32_t srcOffsets[64];
            int32_t tplCount = 0;
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();
            for (int32_t pi = 0; pi < linearN && tplCount < nUS; pi++) {
                int32_t pbScan = (extStart / repeatPeriod + pi) * repeatPeriod;
                int32_t pSrc = SrcOff(pbScan);
                bool found = false;
                for (int32_t tt = 0; tt < tplCount; tt++) {
                    if (srcOffsets[tt] == pSrc) {
                        found = true;
                        break;
                    }
                }
                if (found)
                    continue;
                srcOffsets[tplCount] = pSrc;
                LocalTensor<T> inBuf = inQue.AllocTensor<T>();
                if (seedBulkOk) {
                    int32_t ba = ((seedInElems + alignElems - 1) / alignElems) * alignElems;
                    CopyIn(inBuf, gmIn[pSrc], ba);
                } else {
                    for (int32_t s = 0; s < repeatInputPeriod; s++)
                        CopyIn(inBuf[s * innerDimAligned], gmIn[SrcOff(pbScan + s)], innerDimAligned);
                }
                inQue.EnQue(inBuf);
                LocalTensor<T> inData = inQue.DeQue<T>();
                int32_t tplBase = tplCount * seedOutAligned;
                if (innerMult == 1) {
                    if (seedBulkOk) {
                        int32_t ca = ((seedOutElems + alignElems - 1) / alignElems) * alignElems;
                        DataCopy(outBuf[tplBase], inData, ca);
                    } else {
                        for (int32_t bIdx = 0; bIdx < repeatInputPeriod; bIdx++)
                            DataCopy(
                                outBuf[tplBase + bIdx * outputInnerDim], inData[bIdx * seedSrcStride], innerDimAligned);
                    }
                } else {
                    for (int32_t bIdx = 0; bIdx < repeatInputPeriod; bIdx++) {
                        int32_t srcBase = bIdx * seedSrcStride;
                        int32_t dstBase = tplBase + bIdx * outputInnerDim;
                        BuildInnerRow(outBuf, inData, srcBase, dstBase, innerDim, innerMult);
                    }
                }
                inQue.FreeTensor(inData);
                tplCount++;
            }
            outQue.EnQue(outBuf);
            LocalTensor<T> outData = outQue.DeQue<T>();
            int32_t fullPeriodElems = repeatPeriod * outputInnerDim;
            int32_t fullPeriodAligned = ((fullPeriodElems + alignElems - 1) / alignElems) * alignElems;
            bool canBroadcastPeriod =
                (numAmplify >= 3 && fullPeriodAligned <= bufElems && tplCount <= 8 && linearN > tplCount * 5 &&
                 linearN * numAmplify >= 500);
            if (canBroadcastPeriod) {
                int32_t repBases[64];
                for (int32_t tt = 0; tt < tplCount; tt++)
                    repBases[tt] = -1;
                for (int32_t pi = 0; pi < linearN; pi++) {
                    int32_t pb = (extStart / repeatPeriod + pi) * repeatPeriod;
                    if (pb < myStartRow || pb + repeatPeriod > myEndRow)
                        continue;
                    int32_t pSrc = SrcOff(pb);
                    int32_t tplIdx = 0;
                    for (int32_t tt = 0; tt < tplCount; tt++) {
                        if (srcOffsets[tt] == pSrc) {
                            tplIdx = tt;
                            break;
                        }
                    }
                    if (repBases[tplIdx] >= 0)
                        continue;
                    repBases[tplIdx] = pb;
                    int32_t tplOff = tplIdx * seedOutAligned;
                    for (int32_t p = 0; p < numAmplify; p++) {
                        int32_t destRow = pb + p * repeatInputPeriod;
                        CopyOut(gmOut[destRow * outputInnerDim], outData[tplOff], seedOutElems);
                    }
                }
                WaitMTE3();
                outQue.FreeTensor(outData);
                bool allFound = true;
                for (int32_t tt = 0; tt < tplCount; tt++) {
                    if (repBases[tt] < 0) {
                        allFound = false;
                        break;
                    }
                }
                if (allFound) {
                    for (int32_t tt = 0; tt < tplCount; tt++) {
                        LocalTensor<T> rpBuf = inQue.AllocTensor<T>();
                        CopyIn(rpBuf, gmOut[repBases[tt] * outputInnerDim], fullPeriodAligned);
                        event_t rdEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                        SetFlag<HardEvent::MTE2_MTE3>(rdEv);
                        WaitFlag<HardEvent::MTE2_MTE3>(rdEv);
                        inQue.EnQue(rpBuf);
                        LocalTensor<T> rpData = inQue.DeQue<T>();
                        for (int32_t pi = 0; pi < linearN; pi++) {
                            int32_t pb = (extStart / repeatPeriod + pi) * repeatPeriod;
                            int32_t pSrc = SrcOff(pb);
                            int32_t tIdx = 0;
                            for (int32_t t2 = 0; t2 < tplCount; t2++) {
                                if (srcOffsets[t2] == pSrc) {
                                    tIdx = t2;
                                    break;
                                }
                            }
                            if (tIdx != tt)
                                continue;
                            if (pb == repBases[tt])
                                continue;
                            int32_t destRow = pb;
                            if (destRow >= myEndRow)
                                continue;
                            int32_t wrElems = fullPeriodElems;
                            if (destRow + repeatPeriod > myEndRow) {
                                wrElems = (myEndRow - destRow) * outputInnerDim;
                            }
                            if (destRow < myStartRow) {
                                int32_t skip = myStartRow - destRow;
                                int32_t srcOff = skip * outputInnerDim;
                                wrElems -= srcOff;
                                if (wrElems <= 0)
                                    continue;
                                CopyOut(gmOut[myStartRow * outputInnerDim], rpData[srcOff], wrElems);
                            } else {
                                CopyOut(gmOut[destRow * outputInnerDim], rpData, wrElems);
                            }
                        }
                        WaitMTE3();
                        inQue.FreeTensor(rpData);
                    }
                } else {
                    LocalTensor<T> outData2 = outQue.AllocTensor<T>();
                    for (int32_t tt = 0; tt < tplCount; tt++) {
                        if (repBases[tt] >= 0)
                            continue;
                        int32_t tplOff = tt * seedOutAligned;
                        for (int32_t pi = 0; pi < linearN; pi++) {
                            int32_t pb = (extStart / repeatPeriod + pi) * repeatPeriod;
                            int32_t pSrc = SrcOff(pb);
                            int32_t tIdx = 0;
                            for (int32_t t2 = 0; t2 < tplCount; t2++) {
                                if (srcOffsets[t2] == pSrc) {
                                    tIdx = t2;
                                    break;
                                }
                            }
                            if (tIdx != tt)
                                continue;
                            for (int32_t p = 0; p < numAmplify; p++) {
                                int32_t destRow = pb + p * repeatInputPeriod;
                                if (destRow >= myEndRow)
                                    break;
                                if (destRow < myStartRow)
                                    continue;
                                int32_t wr = repeatInputPeriod;
                                if (destRow + wr > myEndRow)
                                    wr = myEndRow - destRow;
                                if (wr <= 0)
                                    continue;
                                CopyOut(gmOut[destRow * outputInnerDim], outData2[tplOff], wr * outputInnerDim);
                            }
                        }
                    }
                    WaitMTE3();
                    outQue.FreeTensor(outData2);
                }
            } else {
                for (int32_t pi = 0; pi < linearN; pi++) {
                    int32_t pb = (extStart / repeatPeriod + pi) * repeatPeriod;
                    int32_t pSrc = SrcOff(pb);
                    int32_t tplIdx = 0;
                    for (int32_t tt = 0; tt < tplCount; tt++) {
                        if (srcOffsets[tt] == pSrc) {
                            tplIdx = tt;
                            break;
                        }
                    }
                    int32_t tplOff = tplIdx * seedOutAligned;
                    for (int32_t p = 0; p < numAmplify; p++) {
                        int32_t destRow = pb + p * repeatInputPeriod;
                        if (destRow >= myEndRow)
                            break;
                        if (destRow < myStartRow)
                            continue;
                        int32_t wr = repeatInputPeriod;
                        if (destRow + wr > myEndRow)
                            wr = myEndRow - destRow;
                        if (wr <= 0)
                            continue;
                        CopyOut(gmOut[destRow * outputInnerDim], outData[tplOff], wr * outputInnerDim);
                    }
                }
                WaitMTE3();
                outQue.FreeTensor(outData);
            }
            if (lastFullPeriodEnd < myEndRow) {
                myStartRow = lastFullPeriodEnd;
                ProcessScalarBuildFast(innerDimAligned, bufElems);
            }
            return;
        }
        int32_t curTemplateSrc = -1;
        LocalTensor<T> outData;
        bool templateBuilt = false;
        int32_t loopN = srcGrp ? (periodsPS * srcPerCore) : ((extEnd - extStart) / repeatPeriod);
        for (int32_t si = 0; si < loopN; si++) {
            int32_t pb;
            if (srcGrp) {
                int32_t localSrc = si / periodsPS;
                int32_t srcPeriod = si % periodsPS;
                int32_t globalSrc = GetBlockIdx() * srcPerCore + localSrc;
                if (globalSrc >= nSrc)
                    break;
                pb = (srcPeriod * nSrc + globalSrc) * repeatPeriod;
            } else {
                pb = extStart + si * repeatPeriod;
            }
            int32_t periodSrc = SrcOff(pb);
            if (periodSrc != curTemplateSrc) {
                if (templateBuilt) {
                    WaitMTE3();
                    outQue.FreeTensor(outData);
                }
                LocalTensor<T> inBuf = inQue.AllocTensor<T>();
                if (seedBulkOk) {
                    int32_t ba = ((seedInElems + alignElems - 1) / alignElems) * alignElems;
                    CopyIn(inBuf, gmIn[periodSrc], ba);
                } else {
                    for (int32_t s = 0; s < repeatInputPeriod; s++)
                        CopyIn(inBuf[s * innerDimAligned], gmIn[SrcOff(pb + s)], innerDimAligned);
                }
                inQue.EnQue(inBuf);
                LocalTensor<T> inData = inQue.DeQue<T>();
                LocalTensor<T> outBuf = outQue.AllocTensor<T>();
                if (innerMult == 1) {
                    int32_t copyElems = repeatInputPeriod * innerDim;
                    int32_t copyAligned = ((copyElems + alignElems - 1) / alignElems) * alignElems;
                    if (seedBulkOk && copyAligned <= bufElems) {
                        DataCopy(outBuf, inData, copyAligned);
                    } else {
                        for (int32_t bIdx = 0; bIdx < repeatInputPeriod; bIdx++) {
                            DataCopy(outBuf[bIdx * outputInnerDim], inData[bIdx * seedSrcStride], innerDimAligned);
                        }
                    }
                } else {
                    seedOutputOffset = 0;
                    bool useGather = false;
                    if constexpr (sizeof(T) >= 2) {
                        int32_t sRE_check = outputInnerDim;
                        int32_t offRowBytes = outputInnerDim * static_cast<int32_t>(sizeof(T));
                        int32_t gv = offRowBytes;
                        int32_t bv2 = 32;
                        while (bv2) {
                            int32_t tv = bv2;
                            bv2 = gv % bv2;
                            gv = tv;
                        }
                        int32_t sRC = 32 / gv;
                        if (sRC <= repeatInputPeriod)
                            sRE_check = sRC * outputInnerDim;
                        int32_t offU32 = ((sRE_check + 7) / 8) * 8;
                        int32_t offT =
                            (offU32 * 4 + static_cast<int32_t>(sizeof(T)) - 1) / static_cast<int32_t>(sizeof(T));
                        int32_t gStart = ((offT + alignElems - 1) / alignElems) * alignElems;
                        bool gatherFits = (gStart + seedOutElems <= bufElems);
                        bool superRowOk = (sRC >= 2 && repeatInputPeriod >= 4 * sRC);
                        bool perRowOk = (sRC < repeatInputPeriod && repeatInputPeriod >= 4);
                        int32_t tableElems = (sRC <= repeatInputPeriod) ? sRC * outputInnerDim : outputInnerDim;
                        int32_t scalarCost = repeatInputPeriod * outputInnerDim;
                        bool costOk = (tableElems * 5 < scalarCost);
                        if (gatherFits && (superRowOk || (perRowOk && costOk))) {
                            int32_t gOff = GatherBuildRows(
                                outBuf, inData, repeatInputPeriod, innerDim, innerMult, seedSrcStride, outputInnerDim,
                                alignElems);
                            if (gOff >= 0) {
                                seedOutputOffset = gOff;
                                useGather = true;
                            }
                        }
                    }
                    if constexpr (sizeof(T) == 1) {
                        if (!useGather && innerDim % 2 == 0 && outputInnerDim % 2 == 0 && seedSrcStride % 2 == 0 &&
                            repeatInputPeriod >= 2 && (outputInnerDim % DATA_BLOCK_BYTES == 0)) {
                            int32_t oiDim_u16 = outputInnerDim / 2;
                            int32_t offU32_u8 = ((oiDim_u16 + 7) / 8) * 8;
                            int32_t offB = offU32_u8 * 4;
                            int32_t offE16 = (offB + 1) / 2;
                            constexpr int32_t ALN16 = DATA_BLOCK_BYTES / 2;
                            int32_t gS16 = ((offE16 + ALN16 - 1) / ALN16) * ALN16;
                            int32_t gS8 = gS16 * 2;
                            if (gS8 + seedOutElems <= bufElems) {
                                int32_t gOff = GatherBuildRowsU8(
                                    outBuf, inData, repeatInputPeriod, innerDim, innerMult, seedSrcStride,
                                    outputInnerDim, alignElems);
                                if (gOff >= 0) {
                                    seedOutputOffset = gOff;
                                    useGather = true;
                                }
                            }
                        }
                    }
                    if (!useGather) {
                        for (int32_t bIdx = 0; bIdx < repeatInputPeriod; bIdx++) {
                            int32_t srcBase = bIdx * seedSrcStride;
                            int32_t dstBase = bIdx * outputInnerDim;
                            BuildInnerRow(outBuf, inData, srcBase, dstBase, innerDim, innerMult);
                        }
                    }
                } // end else (innerMult > 1)
                inQue.FreeTensor(inData);
                outQue.EnQue(outBuf);
                outData = outQue.DeQue<T>();
                curTemplateSrc = periodSrc;
                templateBuilt = true;
            }
            for (int32_t p = 0; p < numAmplify; p++) {
                int32_t destRow = pb + p * repeatInputPeriod;
                int32_t destEnd = destRow + repeatInputPeriod;
                if (destRow >= myEndRow)
                    break;
                if (destEnd <= myStartRow)
                    continue;
                if (destRow < myStartRow) {
                    int32_t overlapStart = myStartRow;
                    int32_t overlapEnd = (destEnd <= myEndRow) ? destEnd : myEndRow;
                    if (overlapStart < overlapEnd) {
                        int32_t tplOff = overlapStart - destRow;
                        int32_t srcElOff = tplOff * outputInnerDim;
                        int32_t srcByteOff = srcElOff * elemBytes;
                        int32_t dstByteOff = overlapStart * outputInnerDim * elemBytes;
                        if (srcByteOff % DATA_BLOCK_BYTES == 0 && dstByteOff % DATA_BLOCK_BYTES == 0) {
                            int32_t writeRows = overlapEnd - overlapStart;
                            CopyOut(
                                gmOut[overlapStart * outputInnerDim], outData[seedOutputOffset + srcElOff],
                                writeRows * outputInnerDim);
                        } else {
                            if (overlapEnd > scalarFallbackEnd) {
                                scalarFallbackEnd = overlapEnd;
                            }
                        }
                    }
                    continue;
                }
                int32_t wr = (destEnd > myEndRow) ? (myEndRow - destRow) : repeatInputPeriod;
                if (wr <= 0)
                    continue;
                CopyOut(gmOut[destRow * outputInnerDim], outData[seedOutputOffset], wr * outputInnerDim);
            }
        }
        if (templateBuilt) {
            WaitMTE3();
            outQue.FreeTensor(outData);
        }
        if (scalarFallbackStart >= scalarFallbackEnd) {
            if (srcGrp && remainderRows > 0) {
                int32_t lastActiveSrc = nSrc - 1;
                int32_t lastActiveCoreForSrc = lastActiveSrc / srcPerCore;
                if (GetBlockIdx() == lastActiveCoreForSrc) {
                    myStartRow = totalPeriods * repeatPeriod;
                    myEndRow = outerCount;
                    ProcessScalarBuildFast(innerDimAligned, bufElems);
                }
            }
            return;
        }
        myStartRow = scalarFallbackStart;
        myEndRow = scalarFallbackEnd;
    }
    ProcessScalarBuildFast(innerDimAligned, bufElems);
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessDmaTemplateBuild(int32_t innerDimAligned, int32_t bufElems)
{
    int32_t repeatPeriod = tilingRepeatPeriod;
    int32_t repeatInputPeriod = tilingRepeatInputPeriod;
    int32_t numAmplify = repeatPeriod / repeatInputPeriod;
    int32_t seedOutElems = repeatInputPeriod * outputInnerDim;
    int32_t seedInElems = (repeatInputPeriod - 1) * innerDim + innerDimAligned;
    bool seedBulkOk =
        (repeatInputPeriod <= 1) || (SrcOff(1) - SrcOff(0) == innerDim &&
                                     SrcOff(repeatInputPeriod - 1) == SrcOff(0) + (repeatInputPeriod - 1) * innerDim);

    int32_t firstFullPeriod = ((myStartRow + repeatPeriod - 1) / repeatPeriod) * repeatPeriod;
    int32_t lastFullPeriodEnd = (myEndRow / repeatPeriod) * repeatPeriod;
    int32_t extStart = (myStartRow / repeatPeriod) * repeatPeriod;
    int32_t extEnd = lastFullPeriodEnd;
    if (lastFullPeriodEnd < myEndRow) {
        extEnd = lastFullPeriodEnd + repeatPeriod;
    }
    int32_t scalarFallbackStart = myStartRow;
    int32_t scalarFallbackEnd = myStartRow;

    if (extStart < extEnd) {
        int32_t curTemplateSrc = -1;
        for (int32_t pb = extStart; pb < extEnd; pb += repeatPeriod) {
            int32_t periodSrc = SrcOff(pb);
            if (periodSrc != curTemplateSrc) {
                LocalTensor<T> inBuf = inQue.AllocTensor<T>();
                if (seedBulkOk) {
                    for (int32_t s = 0; s < repeatInputPeriod; s++)
                        CopyIn(inBuf[s * innerDimAligned], gmIn[periodSrc + s * innerDim], innerDimAligned);
                } else {
                    for (int32_t s = 0; s < repeatInputPeriod; s++)
                        CopyIn(inBuf[s * innerDimAligned], gmIn[periodSrc + SrcOff(s)], innerDimAligned);
                }
                inQue.EnQue(inBuf);
                LocalTensor<T> inData = inQue.DeQue<T>();
                int32_t destBase = pb * outputInnerDim;
                for (int32_t row = 0; row < repeatInputPeriod; row++) {
                    int32_t srcUb = row * innerDimAligned;
                    int32_t gmRowBase = destBase + row * outputInnerDim;
                    for (int32_t m = 0; m < innerMult; m++) {
                        CopyOut(gmOut[gmRowBase + m * innerDim], inData[srcUb], innerDim);
                    }
                }
                WaitMTE3();
                inQue.FreeTensor(inData);
                curTemplateSrc = periodSrc;
            }
            LocalTensor<T> tplBuf = outQue.AllocTensor<T>();
            int32_t gmTplBase = pb * outputInnerDim;
            CopyIn(tplBuf, gmOut[gmTplBase], seedOutElems);
            event_t rdEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(rdEv);
            WaitFlag<HardEvent::MTE2_MTE3>(rdEv);
            outQue.EnQue(tplBuf);
            LocalTensor<T> outData = outQue.DeQue<T>();
            for (int32_t p = 1; p < numAmplify; p++) {
                int32_t destRow = pb + p * repeatInputPeriod;
                if (destRow >= myEndRow)
                    break;
                if (destRow + repeatInputPeriod <= myStartRow) {
                    if (destRow + repeatInputPeriod > scalarFallbackEnd)
                        scalarFallbackEnd = destRow + repeatInputPeriod;
                    continue;
                }
                if (destRow < myStartRow) {
                    int32_t overlapEnd = destRow + repeatInputPeriod;
                    if (overlapEnd > myEndRow)
                        overlapEnd = myEndRow;
                    if (overlapEnd > scalarFallbackEnd)
                        scalarFallbackEnd = overlapEnd;
                    continue;
                }
                int32_t wr = repeatInputPeriod;
                if (destRow + wr > myEndRow)
                    wr = myEndRow - destRow;
                if (wr <= 0)
                    continue;
                CopyOut(gmOut[destRow * outputInnerDim], outData, wr * outputInnerDim);
            }
            WaitMTE3();
            outQue.FreeTensor(outData);
        }
        if (scalarFallbackStart >= scalarFallbackEnd)
            return;
        myStartRow = scalarFallbackStart;
        myEndRow = scalarFallbackEnd;
    }
    ProcessScalarBuildFast(innerDimAligned, bufElems);
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessRowReadbackAmplify(int32_t innerDimAligned, int32_t bufElems)
{
    int32_t repeatPeriod = tilingRepeatPeriod;
    int32_t repeatInputPeriod = tilingRepeatInputPeriod;
    if (repeatPeriod <= repeatInputPeriod || repeatInputPeriod <= 0) {
        ProcessScalarBuildFast(innerDimAligned, bufElems);
        return;
    }
    int32_t numAmplify = repeatPeriod / repeatInputPeriod;

    int32_t numPeriods = outerCount / repeatPeriod;
    int32_t totalSeeds = numPeriods * repeatInputPeriod;
    int32_t remainder = outerCount - numPeriods * repeatPeriod;
    if (remainder > 0) {
        int32_t extraSeeds = (remainder < repeatInputPeriod) ? remainder : repeatInputPeriod;
        totalSeeds += extraSeeds;
    }

    int32_t seedsPerCore = (totalSeeds + blockDimVal - 1) / blockDimVal;
    int32_t coreId = GetBlockIdx();
    int32_t mySeedStart = coreId * seedsPerCore;
    int32_t mySeedEnd = mySeedStart + seedsPerCore;
    if (mySeedEnd > totalSeeds)
        mySeedEnd = totalSeeds;
    if (mySeedStart >= totalSeeds)
        return;

    int32_t maxBatchByIn = bufElems / innerDimAligned;
    int32_t maxBatchByOut = bufElems / outputInnerDim;
    bool dmaBuildPossible = (innerMult > 1 && innerDimAligned * innerMult <= bufElems && innerDim * elemBytes >= 32);

    bool useGatherHybrid = false;
    int32_t ghStart = 0;
    int32_t ghRowStride = 0;
    int32_t ghMaxBatch = 0;
    if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
        if (dmaBuildPossible && innerDim != innerDimAligned) {
            constexpr int32_t VREG_GH = 256 / static_cast<int32_t>(sizeof(T));
            ghRowStride =
                (outputInnerDim > VREG_GH) ? (((outputInnerDim + alignElems - 1) / alignElems) * alignElems) : VREG_GH;
            int32_t offU32 = ((outputInnerDim + 7) / 8) * 8;
            int32_t offT = (offU32 * 4 + static_cast<int32_t>(sizeof(T)) - 1) / static_cast<int32_t>(sizeof(T));
            ghStart = ((offT + alignElems - 1) / alignElems) * alignElems;
            ghMaxBatch = (bufElems - ghStart) / ghRowStride;
            if (ghMaxBatch > maxBatchByOut)
                ghMaxBatch = maxBatchByOut;
            if (ghMaxBatch > maxBatchByIn)
                ghMaxBatch = maxBatchByIn;
            if (ghMaxBatch > repeatInputPeriod)
                ghMaxBatch = repeatInputPeriod;
            if (ghMaxBatch >= 1) {
                int32_t totalSeedsEst = outerCount / (repeatPeriod > 0 ? repeatPeriod : 1) *
                                        (repeatInputPeriod > 0 ? repeatInputPeriod : outerCount);
                int32_t seedsPerCoreEst = (totalSeedsEst + blockDimVal - 1) / blockDimVal;
                if (seedsPerCoreEst >= 50)
                    useGatherHybrid = true;
            }
        }
    }

    if (useGatherHybrid) {
        int32_t si = mySeedStart;
        LocalTensor<T> gOutBuf = outQue.AllocTensor<T>();
        auto gOff = gOutBuf[0].template ReinterpretCast<uint32_t>();
        for (int32_t k = 0; k < outputInnerDim; k++) {
            gOff.SetValue(k, static_cast<uint32_t>((k % innerDim) * static_cast<int32_t>(sizeof(T))));
        }
        PipeBarrier<PIPE_V>();

        while (si < mySeedEnd) {
            int32_t periodIdx = si / repeatInputPeriod;
            int32_t posStart = si % repeatInputPeriod;
            int32_t periodBase = periodIdx * repeatPeriod;
            int32_t seedBase = periodBase + posStart;
            int32_t remainInPeriod = repeatInputPeriod - posStart;
            int32_t remainTotal = mySeedEnd - si;
            int32_t bc = ghMaxBatch;
            if (bc > remainInPeriod)
                bc = remainInPeriod;
            if (bc > remainTotal)
                bc = remainTotal;

            LocalTensor<T> inBuf = inQue.AllocTensor<T>();
            for (int32_t s = 0; s < bc; s++)
                CopyIn(inBuf[s * innerDimAligned], gmIn[SrcOff(seedBase + s)], innerDimAligned);
            inQue.EnQue(inBuf);
            LocalTensor<T> inData = inQue.DeQue<T>();

            for (int32_t r = 0; r < bc; r++) {
                Gather(
                    gOutBuf[ghStart + r * ghRowStride], inData[r * innerDimAligned], gOff, static_cast<uint32_t>(0),
                    static_cast<uint32_t>(outputInnerDim));
            }
            PipeBarrier<PIPE_V>();
            ResetMask();
            inQue.FreeTensor(inData);

            event_t vToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToM);
            WaitFlag<HardEvent::V_MTE3>(vToM);
            for (int32_t r = 0; r < bc; r++) {
                CopyOut(gmOut[(seedBase + r) * outputInnerDim], gOutBuf[ghStart + r * ghRowStride], outputInnerDim);
            }
            event_t mToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(mToV);
            WaitFlag<HardEvent::MTE3_V>(mToV);
            WaitMTE3();

            LocalTensor<T> ampBuf = inQue.AllocTensor<T>();
            CopyIn(ampBuf, gmOut[seedBase * outputInnerDim], bc * outputInnerDim);
            event_t rdEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(rdEv);
            WaitFlag<HardEvent::MTE2_MTE3>(rdEv);
            inQue.EnQue(ampBuf);
            LocalTensor<T> ampData = inQue.DeQue<T>();
            for (int32_t p = 1; p < numAmplify; p++) {
                int32_t ampRow = periodBase + p * repeatInputPeriod + posStart;
                if (ampRow >= outerCount)
                    break;
                int32_t ampEnd = ampRow + bc;
                if (ampEnd > outerCount) {
                    CopyOut(gmOut[ampRow * outputInnerDim], ampData, (outerCount - ampRow) * outputInnerDim);
                } else {
                    CopyOut(gmOut[ampRow * outputInnerDim], ampData, bc * outputInnerDim);
                }
            }
            WaitMTE3();
            inQue.FreeTensor(ampData);
            si += bc;
        }
        outQue.FreeTensor(gOutBuf);
        return;
    }
    if (dmaBuildPossible) {
        int32_t maxBatchByDma = bufElems / (innerMult * innerDimAligned);
        if (maxBatchByDma < maxBatchByOut)
            maxBatchByOut = maxBatchByDma;
    }
    int32_t maxBatch = (maxBatchByIn < maxBatchByOut) ? maxBatchByIn : maxBatchByOut;
    if (maxBatch > repeatInputPeriod)
        maxBatch = repeatInputPeriod;
    if (maxBatch < 1)
        maxBatch = 1;

    int32_t si = mySeedStart;
    while (si < mySeedEnd) {
        int32_t periodIdx = si / repeatInputPeriod;
        int32_t posStart = si % repeatInputPeriod;
        int32_t periodBase = periodIdx * repeatPeriod;
        int32_t seedBase = periodBase + posStart;

        int32_t remainInPeriod = repeatInputPeriod - posStart;
        int32_t remainTotal = mySeedEnd - si;
        int32_t bc = maxBatch;
        if (bc > remainInPeriod)
            bc = remainInPeriod;
        if (bc > remainTotal)
            bc = remainTotal;

        bool useDmaBuild = (innerMult > 1 && innerDimAligned * innerMult <= bufElems && innerDim * elemBytes >= 32);
        LocalTensor<T> inBuf = inQue.AllocTensor<T>();
        bool seedBulk =
            (!useDmaBuild && bc > 1 && SrcOff(seedBase + 1) - SrcOff(seedBase) == innerDim &&
             SrcOff(seedBase + bc - 1) == SrcOff(seedBase) + (bc - 1) * innerDim);
        if (seedBulk) {
            int32_t bulkLen = (bc - 1) * innerDim + innerDimAligned;
            int32_t bulkAligned = ((bulkLen + alignElems - 1) / alignElems) * alignElems;
            CopyIn(inBuf, gmIn[SrcOff(seedBase)], bulkAligned);
        } else {
            for (int32_t s = 0; s < bc; s++)
                CopyIn(inBuf[s * innerDimAligned], gmIn[SrcOff(seedBase + s)], innerDimAligned);
        }
        inQue.EnQue(inBuf);
        LocalTensor<T> inData = inQue.DeQue<T>();

        if (useDmaBuild) {
            bool useGatherWholeRow = false;
            if (useGatherWholeRow) {
                int32_t oiDim = outputInnerDim;
                int32_t offU32G = ((oiDim + 7) / 8) * 8;
                int32_t offTG = (offU32G * 4 + static_cast<int32_t>(sizeof(T)) - 1) / static_cast<int32_t>(sizeof(T));
                int32_t gStartG = ((offTG + alignElems - 1) / alignElems) * alignElems;
                LocalTensor<T> outBuf = outQue.AllocTensor<T>();
                auto gOff = outBuf[0].template ReinterpretCast<uint32_t>();
                for (int32_t k = 0; k < oiDim; k++) {
                    gOff.SetValue(k, static_cast<uint32_t>((k % innerDim) * static_cast<int32_t>(sizeof(T))));
                }
                PipeBarrier<PIPE_V>();
                for (int32_t r = 0; r < bc; r++) {
                    Gather(
                        outBuf[gStartG], inData[r * innerDimAligned], gOff, static_cast<uint32_t>(0),
                        static_cast<uint32_t>(oiDim));
                    PipeBarrier<PIPE_V>();
                    ResetMask();
                    int32_t gmBase = (seedBase + r) * outputInnerDim;
                    event_t vToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(vToM);
                    WaitFlag<HardEvent::V_MTE3>(vToM);
                    CopyOut(gmOut[gmBase], outBuf[gStartG], oiDim);
                    event_t mToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                    SetFlag<HardEvent::MTE3_V>(mToV);
                    WaitFlag<HardEvent::MTE3_V>(mToV);
                }
                inQue.FreeTensor(inData);
                outQue.FreeTensor(outBuf);
                LocalTensor<T> ampBuf = inQue.AllocTensor<T>();
                int32_t seedGmOff = seedBase * outputInnerDim;
                CopyIn(ampBuf, gmOut[seedGmOff], bc * outputInnerDim);
                event_t rdEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(rdEv);
                WaitFlag<HardEvent::MTE2_MTE3>(rdEv);
                inQue.EnQue(ampBuf);
                LocalTensor<T> ampData = inQue.DeQue<T>();
                for (int32_t p = 1; p < numAmplify; p++) {
                    int32_t ampRow = periodBase + p * repeatInputPeriod + posStart;
                    if (ampRow >= outerCount)
                        break;
                    int32_t ampEnd = ampRow + bc;
                    if (ampEnd > outerCount) {
                        CopyOut(gmOut[ampRow * outputInnerDim], ampData, (outerCount - ampRow) * outputInnerDim);
                    } else {
                        CopyOut(gmOut[ampRow * outputInnerDim], ampData, bc * outputInnerDim);
                    }
                }
                WaitMTE3();
                inQue.FreeTensor(ampData);
            } else {
                {
                    event_t rdDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                    SetFlag<HardEvent::MTE2_MTE3>(rdDone);
                    WaitFlag<HardEvent::MTE2_MTE3>(rdDone);
                }
                for (int32_t r = 0; r < bc; r++) {
                    int32_t gmBase = (seedBase + r) * outputInnerDim;
                    for (int32_t m = 0; m < innerMult; m++) {
                        CopyOut(gmOut[gmBase + m * innerDim], inData[r * innerDimAligned], innerDim);
                    }
                }
                WaitMTE3();
                inQue.FreeTensor(inData);
                LocalTensor<T> ampBuf = inQue.AllocTensor<T>();
                int32_t seedGmOff = seedBase * outputInnerDim;
                CopyIn(ampBuf, gmOut[seedGmOff], bc * outputInnerDim);
                event_t rdEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(rdEv);
                WaitFlag<HardEvent::MTE2_MTE3>(rdEv);
                inQue.EnQue(ampBuf);
                LocalTensor<T> ampData = inQue.DeQue<T>();
                for (int32_t p = 1; p < numAmplify; p++) {
                    int32_t ampRow = periodBase + p * repeatInputPeriod + posStart;
                    if (ampRow >= outerCount)
                        break;
                    int32_t ampEnd = ampRow + bc;
                    if (ampEnd > outerCount) {
                        int32_t partRows = outerCount - ampRow;
                        CopyOut(gmOut[ampRow * outputInnerDim], ampData, partRows * outputInnerDim);
                    } else {
                        CopyOut(gmOut[ampRow * outputInnerDim], ampData, bc * outputInnerDim);
                    }
                }
                WaitMTE3();
                inQue.FreeTensor(ampData);
            }
        } else {
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();
            int32_t seedSrcStride = seedBulk ? innerDim : innerDimAligned;
            for (int32_t r = 0; r < bc; r++) {
                int32_t srcBase = r * seedSrcStride;
                int32_t dstBase = r * outputInnerDim;
                BuildInnerRow(outBuf, inData, srcBase, dstBase, innerDim, innerMult);
            }
            inQue.FreeTensor(inData);
            outQue.EnQue(outBuf);
            LocalTensor<T> outData = outQue.DeQue<T>();

            CopyOut(gmOut[seedBase * outputInnerDim], outData, bc * outputInnerDim);
            WaitMTE3();

            for (int32_t p = 1; p < numAmplify; p++) {
                int32_t ampRow = periodBase + p * repeatInputPeriod + posStart;
                if (ampRow >= outerCount)
                    break;
                int32_t ampEnd = ampRow + bc;
                if (ampEnd > outerCount) {
                    int32_t partRows = outerCount - ampRow;
                    CopyOut(gmOut[ampRow * outputInnerDim], outData, partRows * outputInnerDim);
                } else {
                    CopyOut(gmOut[ampRow * outputInnerDim], outData, bc * outputInnerDim);
                }
            }
            WaitMTE3();
            outQue.FreeTensor(outData);
        } // end else (!useDmaBuild)
        si += bc;
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessScalarBuildFast(int32_t innerDimAligned, int32_t bufElems)
{
    int32_t repeatPeriod = tilingRepeatPeriod;
    int32_t repeatInputPeriod = tilingRepeatInputPeriod;
    bool hasRepeat = (repeatPeriod > repeatInputPeriod && repeatInputPeriod > 0);
    int32_t numAmplify = hasRepeat ? (repeatPeriod / repeatInputPeriod) : 1;

    int32_t maxBatchIn = bufElems / innerDimAligned;
    int32_t maxBatchOut = bufElems / outputInnerDim;
    int32_t maxBatch = (maxBatchIn < maxBatchOut) ? maxBatchIn : maxBatchOut;
    if (maxBatch < 1) {
        maxBatch = 1;
    }
    int32_t rowBytes = outputInnerDim * elemBytes;
    if (hasRepeat && rowBytes > 0 && rowBytes % DATA_BLOCK_BYTES != 0) {
        bool startAligned = (myStartRow * rowBytes % DATA_BLOCK_BYTES == 0);
        if (startAligned) {
            int32_t g = rowBytes;
            int32_t bv = DATA_BLOCK_BYTES;
            while (bv) {
                int32_t t = bv;
                bv = g % bv;
                g = t;
            }
            int32_t alignStep = DATA_BLOCK_BYTES / g;
            if (alignStep > 1 && maxBatch >= alignStep) {
                maxBatch = (maxBatch / alignStep) * alignStep;
            }
        }
    }
    int32_t coordSrcOff = 0;
    int32_t coordOut[MAX_DIM];
    int32_t coordIn[MAX_DIM];
    {
        int32_t temp = myStartRow;
        for (int32_t d = numDims - 2; d >= 0; d--) {
            coordOut[d] = temp % outputShape[d];
            temp /= outputShape[d];
            coordIn[d] = coordOut[d] % inputShape[d];
            coordSrcOff += coordIn[d] * inputStrides[d];
        }
    }
    int32_t outer = myStartRow;
    while (outer < myEndRow) {
        int32_t posInBlock = hasRepeat ? (outer % repeatPeriod) : 0;
        int32_t blockBase = hasRepeat ? (outer - posInBlock) : outer;
        bool isAmplified = hasRepeat && (posInBlock >= repeatInputPeriod) && (blockBase >= myStartRow);
        if (isAmplified) {
            int32_t blockEnd = blockBase + repeatPeriod;
            if (blockEnd > myEndRow) {
                blockEnd = myEndRow;
            }
            outer = blockEnd;
            coordSrcOff = 0;
            int32_t temp = outer;
            for (int32_t d = numDims - 2; d >= 0; d--) {
                coordOut[d] = temp % outputShape[d];
                temp /= outputShape[d];
                coordIn[d] = coordOut[d] % inputShape[d];
                coordSrcOff += coordIn[d] * inputStrides[d];
            }
            continue;
        }
        bool isFallbackRepeat =
            hasRepeat && (posInBlock >= repeatInputPeriod) && outer >= myStartRow + repeatInputPeriod;
        int32_t batchSize = myEndRow - outer;
        if (batchSize > maxBatch) {
            batchSize = maxBatch;
        }
        if (isFallbackRepeat) {
            int32_t srcRow = outer - repeatInputPeriod;
            int32_t repeatSrcGm = srcRow * outputInnerDim;
            if (batchSize > repeatInputPeriod) {
                batchSize = repeatInputPeriod;
            }
            int32_t remainInPeriod = repeatPeriod - posInBlock;
            if (batchSize > remainInPeriod) {
                batchSize = remainInPeriod;
            }
            int32_t dstGm = outer * outputInnerDim;
            int32_t copyElems = batchSize * outputInnerDim;
            int32_t bufElemsLocal = bufferBytes / elemBytes;
            LocalTensor<T> cpBuf = inQue.AllocTensor<T>();
            int32_t copied = 0;
            while (copied < copyElems) {
                int32_t chunk = copyElems - copied;
                if (chunk > bufElemsLocal) {
                    chunk = (bufElemsLocal / alignElems) * alignElems;
                }
                CopyIn(cpBuf, gmOut[repeatSrcGm + copied], chunk);
                event_t rdId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(rdId);
                WaitFlag<HardEvent::MTE2_MTE3>(rdId);
                CopyOut(gmOut[dstGm + copied], cpBuf, chunk);
                event_t wrId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(wrId);
                WaitFlag<HardEvent::MTE3_MTE2>(wrId);
                copied += chunk;
            }
            inQue.FreeTensor(cpBuf);
            outer += batchSize;
            coordSrcOff = 0;
            int32_t tempFR = outer;
            for (int32_t d = numDims - 2; d >= 0; d--) {
                coordOut[d] = tempFR % outputShape[d];
                tempFR /= outputShape[d];
                coordIn[d] = coordOut[d] % inputShape[d];
                coordSrcOff += coordIn[d] * inputStrides[d];
            }
            continue;
        }
        if (hasRepeat) {
            int32_t seedEnd = blockBase + repeatInputPeriod;
            if (seedEnd > outer && outer + batchSize > seedEnd) {
                batchSize = seedEnd - outer;
            } else if (seedEnd <= outer) {
                int32_t periodEnd = blockBase + repeatPeriod;
                if (outer + batchSize > periodEnd) {
                    batchSize = periodEnd - outer;
                }
            }
        }
        if (batchSize < 1) {
            batchSize = 1;
        }
        LocalTensor<T> inBuf = inQue.AllocTensor<T>();
        bool canBulkRead = true;
        if (batchSize > 1) {
            int32_t prevOff = coordSrcOff;
            int32_t tmpCoordOut[MAX_DIM];
            int32_t tmpCoordIn[MAX_DIM];
            int32_t tmpSrcOff = coordSrcOff;
            for (int32_t d = numDims - 2; d >= 0; d--) {
                tmpCoordOut[d] = coordOut[d];
                tmpCoordIn[d] = coordIn[d];
            }
            for (int32_t d = numDims - 2; d >= 0; d--) {
                tmpCoordOut[d]++;
                if (tmpCoordOut[d] < outputShape[d]) {
                    tmpCoordIn[d]++;
                    if (tmpCoordIn[d] >= inputShape[d]) {
                        tmpSrcOff -= (inputShape[d] - 1) * inputStrides[d];
                        tmpCoordIn[d] = 0;
                    } else {
                        tmpSrcOff += inputStrides[d];
                    }
                    break;
                }
                tmpSrcOff -= tmpCoordIn[d] * inputStrides[d];
                tmpCoordOut[d] = 0;
                tmpCoordIn[d] = 0;
            }
            canBulkRead = (tmpSrcOff - prevOff == innerDim);
            if (canBulkRead && batchSize > 2) {
                canBulkRead = (coordOut[numDims - 2] + batchSize - 1 < outputShape[numDims - 2]) &&
                              (coordIn[numDims - 2] + batchSize - 1 < inputShape[numDims - 2]);
            }
        }
        if (canBulkRead && batchSize > 1) {
            int32_t bulkLen = (batchSize - 1) * innerDim + innerDimAligned;
            int32_t bulkAligned = ((bulkLen + alignElems - 1) / alignElems) * alignElems;
            CopyIn(inBuf, gmIn[coordSrcOff], bulkAligned);
        } else if (batchSize == 1) {
            CopyIn(inBuf, gmIn[coordSrcOff], innerDimAligned);
        } else {
            int32_t tmpCoordOut2[MAX_DIM];
            int32_t tmpCoordIn2[MAX_DIM];
            int32_t tmpSrcOff2 = coordSrcOff;
            for (int32_t d = numDims - 2; d >= 0; d--) {
                tmpCoordOut2[d] = coordOut[d];
                tmpCoordIn2[d] = coordIn[d];
            }
            CopyIn(inBuf, gmIn[tmpSrcOff2], innerDimAligned);
            for (int32_t bIdx = 1; bIdx < batchSize; bIdx++) {
                for (int32_t d = numDims - 2; d >= 0; d--) {
                    tmpCoordOut2[d]++;
                    if (tmpCoordOut2[d] < outputShape[d]) {
                        tmpCoordIn2[d]++;
                        if (tmpCoordIn2[d] >= inputShape[d]) {
                            tmpSrcOff2 -= (inputShape[d] - 1) * inputStrides[d];
                            tmpCoordIn2[d] = 0;
                        } else {
                            tmpSrcOff2 += inputStrides[d];
                        }
                        break;
                    }
                    tmpSrcOff2 -= tmpCoordIn2[d] * inputStrides[d];
                    tmpCoordOut2[d] = 0;
                    tmpCoordIn2[d] = 0;
                }
                CopyIn(inBuf[bIdx * innerDimAligned], gmIn[tmpSrcOff2], innerDimAligned);
            }
        }
        inQue.EnQue(inBuf);
        LocalTensor<T> inData = inQue.DeQue<T>();
        int32_t srcStride = canBulkRead ? innerDim : innerDimAligned;
        LocalTensor<T> outBuf = outQue.AllocTensor<T>();
        bool useVecRepeat = false;
        if constexpr (sizeof(T) >= 2 && sizeof(T) <= 4) {
            int32_t repStride = static_cast<int32_t>(innerDimAligned * sizeof(T)) / DATA_BLOCK_BYTES;
            useVecRepeat = batchSize == 1 && innerMult > 1 && innerMult <= 255 && repStride > 0 && repStride <= 255 &&
                           (0 % alignElems == 0) && innerDimAligned * innerMult <= bufElems;
            if (useVecRepeat) {
                uint64_t mask = static_cast<uint64_t>(innerDim);
                if constexpr (sizeof(T) == 2) {
                    auto dstI = outBuf.template ReinterpretCast<int16_t>();
                    auto srcI = inData.template ReinterpretCast<int16_t>();
                    Adds(
                        dstI, srcI, static_cast<int16_t>(0), mask, static_cast<uint8_t>(innerMult),
                        UnaryRepeatParams(1, 1, static_cast<uint8_t>(repStride), 0));
                } else {
                    auto dstI = outBuf.template ReinterpretCast<int32_t>();
                    auto srcI = inData.template ReinterpretCast<int32_t>();
                    Adds(
                        dstI, srcI, static_cast<int32_t>(0), mask, static_cast<uint8_t>(innerMult),
                        UnaryRepeatParams(1, 1, static_cast<uint8_t>(repStride), 0));
                }
            }
        }
        if (!useVecRepeat) {
            if (innerMult == 1) {
                int32_t totalCopy = batchSize * innerDim;
                int32_t totalCopyAligned = ((totalCopy + alignElems - 1) / alignElems) * alignElems;
                if (canBulkRead && totalCopyAligned <= bufElems) {
                    DataCopy(outBuf, inData, totalCopyAligned);
                } else {
                    for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                        DataCopy(outBuf[bIdx * outputInnerDim], inData[bIdx * srcStride], innerDimAligned);
                    }
                }
            } else {
                for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                    int32_t srcBase = bIdx * srcStride;
                    int32_t dstBase = bIdx * outputInnerDim;
                    BuildInnerRow(outBuf, inData, srcBase, dstBase, innerDim, innerMult);
                }
            }
        }
        inQue.FreeTensor(inData);
        outQue.EnQue(outBuf);
        LocalTensor<T> outData = outQue.DeQue<T>();
        int32_t periodEnd = hasRepeat ? (blockBase + repeatPeriod) : (myEndRow + 1);
        for (int32_t p = 0; p < numAmplify; p++) {
            int32_t destRow = blockBase + p * repeatInputPeriod + posInBlock;
            if (destRow >= periodEnd) {
                break;
            }
            if (destRow >= myEndRow) {
                break;
            }
            int32_t writeRows = batchSize;
            if (destRow + writeRows > myEndRow) {
                writeRows = myEndRow - destRow;
            }
            if (writeRows <= 0) {
                break;
            }
            if (useVecRepeat) {
                for (int32_t mIdx = 0; mIdx < innerMult; mIdx++) {
                    CopyOut(
                        gmOut[destRow * outputInnerDim + mIdx * innerDim], outData[mIdx * innerDimAligned], innerDim);
                }
            } else {
                int32_t writeElems = writeRows * outputInnerDim;
                CopyOut(gmOut[destRow * outputInnerDim], outData, writeElems);
            }
        }
        WaitMTE3();
        outQue.FreeTensor(outData);
        outer += batchSize;
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessVecGatherBuild(int32_t innerDimAligned, int32_t bufElems)
{
    if (myStartRow >= myEndRow)
        return;
    int32_t repeatPeriod = tilingRepeatPeriod;
    int32_t repeatInputPeriod = tilingRepeatInputPeriod;
    bool hasRepeat = (repeatPeriod > repeatInputPeriod && repeatInputPeriod > 0);
    if (!hasRepeat) {
        ProcessScalarBuildFast(innerDimAligned, bufElems);
        return;
    }

    int32_t numAmplify = repeatPeriod / repeatInputPeriod;
    int32_t seedOutElems = repeatInputPeriod * outputInnerDim;
    constexpr int32_t VREG = 256 / static_cast<int32_t>(sizeof(T));
    int32_t oiDim = outputInnerDim;
    int32_t gatherRowStride = (oiDim > VREG) ? (((oiDim + alignElems - 1) / alignElems) * alignElems) : VREG;
    int32_t offsetU32 = ((oiDim + 7) / 8) * 8;
    int32_t offsetElemsT = (offsetU32 * 4 + static_cast<int32_t>(sizeof(T)) - 1) / static_cast<int32_t>(sizeof(T));
    int32_t offsetAligned = ((offsetElemsT + alignElems - 1) / alignElems) * alignElems;
    int32_t gatherStart = offsetAligned;

    int32_t totalPeriods = outerCount / repeatPeriod;
    int32_t remainderRows = outerCount - totalPeriods * repeatPeriod;
    int32_t coreId = GetBlockIdx();
    int32_t periodsPerCore = (totalPeriods + blockDimVal - 1) / blockDimVal;
    int32_t myFirstPeriod = coreId * periodsPerCore;
    int32_t myLastPeriod = myFirstPeriod + periodsPerCore;
    if (myLastPeriod > totalPeriods)
        myLastPeriod = totalPeriods;
    if (myFirstPeriod >= totalPeriods) {
        if (remainderRows > 0 && coreId == totalPeriods) {
            myStartRow = totalPeriods * repeatPeriod;
            myEndRow = outerCount;
            ProcessScalarBuildFast(innerDimAligned, bufElems);
        }
        return;
    }
    myStartRow = myFirstPeriod * repeatPeriod;
    myEndRow = (myLastPeriod == totalPeriods && remainderRows > 0) ? outerCount : myLastPeriod * repeatPeriod;

    int32_t extStart = myFirstPeriod * repeatPeriod;
    int32_t curTemplateSrc = -1;

    for (int32_t pi = 0; pi < myLastPeriod - myFirstPeriod; pi++) {
        int32_t pb = extStart + pi * repeatPeriod;
        int32_t periodSrc = SrcOff(pb);
        if (periodSrc != curTemplateSrc) {
            LocalTensor<T> inBuf = inQue.AllocTensor<T>();
            for (int32_t s = 0; s < repeatInputPeriod; s++)
                CopyIn(inBuf[s * innerDimAligned], gmIn[periodSrc + s * innerDim], innerDimAligned);
            inQue.EnQue(inBuf);
            LocalTensor<T> inData = inQue.DeQue<T>();
            LocalTensor<T> outBuf = outQue.AllocTensor<T>();

            auto gOff = outBuf[0].template ReinterpretCast<uint32_t>();
            for (int32_t k = 0; k < oiDim; k++)
                gOff.SetValue(k, static_cast<uint32_t>((k % innerDim) * static_cast<int32_t>(sizeof(T))));
            PipeBarrier<PIPE_V>();
            for (int32_t bIdx = 0; bIdx < repeatInputPeriod; bIdx++)
                Gather(
                    outBuf[gatherStart + bIdx * gatherRowStride], inData[bIdx * innerDimAligned], gOff,
                    static_cast<uint32_t>(0), static_cast<uint32_t>(oiDim));
            PipeBarrier<PIPE_V>();
            for (int32_t bIdx = 0; bIdx < repeatInputPeriod; bIdx++)
                CopyOut(gmOut[(pb + bIdx) * oiDim], outBuf[gatherStart + bIdx * gatherRowStride], oiDim);
            WaitMTE3();
            ResetMask();

            inQue.FreeTensor(inData);
            outQue.FreeTensor(outBuf);
            curTemplateSrc = periodSrc;
        }
        if (numAmplify > 1) {
            LocalTensor<T> ab = inQue.AllocTensor<T>();
            int32_t sca = ((seedOutElems + alignElems - 1) / alignElems) * alignElems;
            CopyIn(ab, gmOut[pb * outputInnerDim], sca);
            event_t re = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(re);
            WaitFlag<HardEvent::MTE2_MTE3>(re);
            for (int32_t p2 = 1; p2 < numAmplify; p2++) {
                int32_t dr = pb + p2 * repeatInputPeriod;
                if (dr >= myEndRow)
                    break;
                int32_t w = repeatInputPeriod;
                if (dr + w > myEndRow)
                    w = myEndRow - dr;
                if (w <= 0)
                    continue;
                CopyOut(gmOut[dr * outputInnerDim], ab, w * outputInnerDim);
            }
            WaitMTE3();
            inQue.FreeTensor(ab);
        }
    }
    if (remainderRows > 0 && myLastPeriod == totalPeriods) {
        myStartRow = totalPeriods * repeatPeriod;
        myEndRow = outerCount;
        ProcessScalarBuildFast(innerDimAligned, bufElems);
    }
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::WaitMTE3()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(id);
    WaitFlag<HardEvent::MTE3_MTE2>(id);
}

template <typename T>
__aicore__ inline int32_t TileOpImpl<T>::SrcOff(int32_t outerIdx)
{
    int32_t offset = 0;
    int32_t temp = outerIdx;
    for (int32_t dim = numDims - 2; dim >= 0; dim--) {
        int32_t outCoord = temp % outputShape[dim];
        temp /= outputShape[dim];
        offset += (outCoord % inputShape[dim]) * inputStrides[dim];
    }
    return offset;
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessFlatResident()
{
    int32_t bufElems = bufferBytes / elemBytes;
    int32_t innerDimAligned = ((innerDim + alignElems - 1) / alignElems) * alignElems;
    int32_t outputDimAligned = ((outputInnerDim + alignElems - 1) / alignElems) * alignElems;
    if (outputDimAligned < 1) {
        outputDimAligned = alignElems;
    }
    int32_t inputAligned = ((totalInputElems + alignElems - 1) / alignElems) * alignElems;
    bool isOutputAligned = (outputInnerDim == outputDimAligned);
    bool innerAligned = (innerDim == innerDimAligned);

    bool useScalarFR = !innerAligned && innerMult > 1 && innerDim <= 12;
    bool useCachedFR = !innerAligned && innerMult > 1 && !useScalarFR;
    bool bulkRead = innerAligned || useScalarFR || useCachedFR;

    LocalTensor<T> inBuf = inQue.AllocTensor<T>();
    if (bulkRead) {
        CopyIn(inBuf, gmIn[0], inputAligned);
    } else {
        int32_t seedRowCount = totalInputElems / innerDim;
        for (int32_t s = 0; s < seedRowCount; s++) {
            int32_t gmStart = s * innerDim;
            int32_t readLen = (gmStart + innerDimAligned <= totalInputElems) ?
                                  innerDimAligned :
                                  ((totalInputElems - gmStart + alignElems - 1) / alignElems) * alignElems;
            CopyIn(inBuf[s * innerDimAligned], gmIn[gmStart], readLen);
        }
    }
    inQue.EnQue(inBuf);
    LocalTensor<T> inData = inQue.DeQue<T>();
    int32_t seedRowCountFR = totalInputElems / innerDim;

    int32_t currentSrcOff = 0;
    int32_t outCoords[MAX_DIM];
    int32_t inCoords[MAX_DIM];
    {
        int32_t temp = myStartRow;
        for (int32_t d = numDims - 2; d >= 0; d--) {
            outCoords[d] = temp % outputShape[d];
            temp /= outputShape[d];
            inCoords[d] = outCoords[d] % inputShape[d];
            currentSrcOff += inCoords[d] * inputStrides[d];
        }
    }

    if (useCachedFR) {
        int32_t cacheRowElems = outputDimAligned;

        LocalTensor<T> outBuf = outQue.AllocTensor<T>();
        for (int32_t s = 0; s < seedRowCountFR; s++) {
            int32_t srcUb = bulkRead ? (s * innerDim) : (s * innerDimAligned);
            int32_t dstUb = s * cacheRowElems;
            BuildInnerRow(outBuf, inData, srcUb, dstUb, innerDim, innerMult);
        }
        outQue.EnQue(outBuf);
        LocalTensor<T> outData = outQue.DeQue<T>();

        for (int32_t outer = myStartRow; outer < myEndRow; outer++) {
            int32_t cacheIdx = currentSrcOff / innerDim;
            CopyOut(gmOut[outer * outputInnerDim], outData[cacheIdx * cacheRowElems], outputInnerDim);

            for (int32_t d = numDims - 2; d >= 0; d--) {
                outCoords[d]++;
                if (outCoords[d] < outputShape[d]) {
                    inCoords[d]++;
                    if (inCoords[d] >= inputShape[d]) {
                        currentSrcOff -= (inputShape[d] - 1) * inputStrides[d];
                        inCoords[d] = 0;
                    } else {
                        currentSrcOff += inputStrides[d];
                    }
                    break;
                }
                currentSrcOff -= inCoords[d] * inputStrides[d];
                outCoords[d] = 0;
                inCoords[d] = 0;
            }
        }
        WaitMTE3();
        outQue.FreeTensor(outData);
        inQue.FreeTensor(inData);
        return;
    }

    int32_t effectiveRowElems =
        innerAligned ? outputDimAligned : (useScalarFR ? outputInnerDim : (innerMult * innerDimAligned));
    int32_t overrun = (innerMult > 1 && innerAligned) ? (innerDimAligned - innerDim) : 0;
    int32_t maxBatch = (bufElems - overrun) / effectiveRowElems;
    if (maxBatch < 1) {
        maxBatch = 1;
    }

    int32_t outer = myStartRow;

    while (outer < myEndRow) {
        int32_t batchSize = myEndRow - outer;
        if (batchSize > maxBatch) {
            batchSize = maxBatch;
        }

        LocalTensor<T> outBuf = outQue.AllocTensor<T>();
        for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            int32_t srcOff = bulkRead ? currentSrcOff : ((currentSrcOff / innerDim) * innerDimAligned);
            int32_t outOff = bIdx * effectiveRowElems;
            if (innerAligned) {
                DataCopy(outBuf[outOff], inData[srcOff], innerDimAligned);
                int32_t copies = 1;
                while (copies * 2 <= innerMult) {
                    int32_t cpLen = ((copies * innerDim + alignElems - 1) / alignElems) * alignElems;
                    DataCopy(outBuf[outOff + copies * innerDim], outBuf[outOff], cpLen);
                    copies *= 2;
                }
                if (copies < innerMult) {
                    int32_t remLen = (((innerMult - copies) * innerDim + alignElems - 1) / alignElems) * alignElems;
                    DataCopy(outBuf[outOff + copies * innerDim], outBuf[outOff], remLen);
                }
            } else if (useScalarFR) {
                if (innerDim <= 3) {
                    T fc0 = inData.GetValue(srcOff);
                    T fc1 = (innerDim > 1) ? inData.GetValue(srcOff + 1) : fc0;
                    T fc2 = (innerDim > 2) ? inData.GetValue(srcOff + 2) : fc0;
                    for (int32_t m = 0; m < innerMult; m++) {
                        int32_t dstOff = outOff + m * innerDim;
                        outBuf.SetValue(dstOff, fc0);
                        if (innerDim > 1) {
                            outBuf.SetValue(dstOff + 1, fc1);
                        }
                        if (innerDim > 2) {
                            outBuf.SetValue(dstOff + 2, fc2);
                        }
                    }
                } else if (elemBytes == 2 && (innerDim % 2) == 0) {
                    auto inU32 = inData.template ReinterpretCast<uint32_t>();
                    auto outU32 = outBuf.template ReinterpretCast<uint32_t>();
                    int32_t halfDim = innerDim / 2;
                    uint32_t pk[6];
                    int32_t srcH = srcOff / 2;
                    for (int32_t e = 0; e < halfDim; e++) {
                        pk[e] = inU32.GetValue(srcH + e);
                    }
                    for (int32_t m = 0; m < innerMult; m++) {
                        int32_t d = (outOff + m * innerDim) / 2;
                        for (int32_t e = 0; e < halfDim; e++) {
                            outU32.SetValue(d + e, pk[e]);
                        }
                    }
                } else if (elemBytes == 2 && (innerDim % 2) == 1 && (outputInnerDim % 2) == 0) {
                    auto inU16 = inData.template ReinterpretCast<uint16_t>();
                    auto outU32 = outBuf.template ReinterpretCast<uint32_t>();
                    uint16_t raw[12];
                    for (int32_t e = 0; e < innerDim; e++) {
                        raw[e] = inU16.GetValue(srcOff + e);
                    }
                    uint32_t pat[12];
                    for (int32_t i = 0; i < innerDim; i++) {
                        pat[i] = raw[(2 * i) % innerDim] | (static_cast<uint32_t>(raw[(2 * i + 1) % innerDim]) << 16);
                    }
                    int32_t totalU32 = (innerDim * innerMult) / 2;
                    int32_t dH = outOff / 2;
                    int32_t reps = totalU32 / innerDim;
                    int32_t rem = totalU32 % innerDim;
                    int32_t pos = dH;
                    for (int32_t r = 0; r < reps; r++) {
                        for (int32_t i = 0; i < innerDim; i++) {
                            outU32.SetValue(pos + i, pat[i]);
                        }
                        pos += innerDim;
                    }
                    for (int32_t i = 0; i < rem; i++) {
                        outU32.SetValue(pos + i, pat[i]);
                    }
                } else if (elemBytes == 1 && innerDim >= 4 && (innerDim % 4) == 0) {
                    auto inU32 = inData.template ReinterpretCast<uint32_t>();
                    auto outU32 = outBuf.template ReinterpretCast<uint32_t>();
                    int32_t quartDim = innerDim / 4;
                    uint32_t pk[3];
                    int32_t srcQ = srcOff / 4;
                    for (int32_t e = 0; e < quartDim; e++) {
                        pk[e] = inU32.GetValue(srcQ + e);
                    }
                    for (int32_t m = 0; m < innerMult; m++) {
                        int32_t d = (outOff + m * innerDim) / 4;
                        for (int32_t e = 0; e < quartDim; e++) {
                            outU32.SetValue(d + e, pk[e]);
                        }
                    }
                } else {
                    T fc[12];
                    for (int32_t e = 0; e < innerDim; e++) {
                        fc[e] = inData.GetValue(srcOff + e);
                    }
                    for (int32_t m = 0; m < innerMult; m++) {
                        int32_t dstOff = outOff + m * innerDim;
                        for (int32_t e = 0; e < innerDim; e++) {
                            outBuf.SetValue(dstOff + e, fc[e]);
                        }
                    }
                }
            } else {
                DataCopy(outBuf[outOff], inData[srcOff], innerDimAligned);
                for (int32_t m = 1; m < innerMult; m++) {
                    DataCopy(outBuf[outOff + m * innerDimAligned], inData[srcOff], innerDimAligned);
                }
            }
            for (int32_t d = numDims - 2; d >= 0; d--) {
                outCoords[d]++;
                if (outCoords[d] < outputShape[d]) {
                    inCoords[d]++;
                    if (inCoords[d] >= inputShape[d]) {
                        currentSrcOff -= (inputShape[d] - 1) * inputStrides[d];
                        inCoords[d] = 0;
                    } else {
                        currentSrcOff += inputStrides[d];
                    }
                    break;
                }
                currentSrcOff -= inCoords[d] * inputStrides[d];
                outCoords[d] = 0;
                inCoords[d] = 0;
            }
        }

        outQue.EnQue(outBuf);
        LocalTensor<T> outData = outQue.DeQue<T>();

        if (useScalarFR) {
            CopyOut(gmOut[outer * outputInnerDim], outData, batchSize * outputInnerDim);
        } else if (!innerAligned && innerMult > 1) {
            for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                int32_t gmDst = (outer + bIdx) * outputInnerDim;
                for (int32_t m = 0; m < innerMult; m++) {
                    CopyOut(
                        gmOut[gmDst + m * innerDim], outData[bIdx * effectiveRowElems + m * innerDimAligned], innerDim);
                }
            }
        } else if (isOutputAligned) {
            CopyOut(gmOut[outer * outputInnerDim], outData, batchSize * outputInnerDim);
        } else {
            for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
                CopyOut(gmOut[(outer + bIdx) * outputInnerDim], outData[bIdx * effectiveRowElems], outputInnerDim);
            }
        }

        WaitMTE3();
        outQue.FreeTensor(outData);
        outer += batchSize;
    }

    inQue.FreeTensor(inData);
}

template <typename T>
__aicore__ inline void TileOpImpl<T>::ProcessLargeInner(int32_t bufElems)
{
    int32_t chunkAligned = (bufElems / alignElems) * alignElems;
    if (chunkAligned < alignElems) {
        chunkAligned = alignElems;
    }
    int32_t numChunks = (innerDim + chunkAligned - 1) / chunkAligned;
    int32_t totalUnits = outerCount * numChunks;
    int32_t coreId = GetBlockIdx();
    int32_t unitsPerCore = (totalUnits + blockDimVal - 1) / blockDimVal;
    int32_t myUnitStart = coreId * unitsPerCore;
    int32_t myUnitEnd = myUnitStart + unitsPerCore;
    if (myUnitEnd > totalUnits) {
        myUnitEnd = totalUnits;
    }
    if (myUnitStart >= totalUnits) {
        return;
    }
    for (int32_t unit = myUnitStart; unit < myUnitEnd; unit++) {
        int32_t outer = unit / numChunks;
        int32_t chunkIdx = unit % numChunks;
        int32_t srcOff = SrcOff(outer);
        int32_t dstBase = outer * outputInnerDim;
        int32_t copied = chunkIdx * chunkAligned;
        int32_t remain = innerDim - copied;
        int32_t chunk = (remain >= chunkAligned) ? chunkAligned : remain;
        int32_t copySize = ((chunk + alignElems - 1) / alignElems) * alignElems;
        LocalTensor<T> buf = inQue.AllocTensor<T>();
        CopyIn(buf, gmIn[srcOff + copied], copySize);
        inQue.EnQue(buf);
        LocalTensor<T> bufData = inQue.DeQue<T>();
        LocalTensor<T> outBuf = outQue.AllocTensor<T>();
        DataCopy(outBuf, bufData, copySize);
        inQue.FreeTensor(bufData);
        outQue.EnQue(outBuf);
        LocalTensor<T> outData = outQue.DeQue<T>();
        for (int32_t mIdx = 0; mIdx < innerMult; mIdx++) {
            int32_t dstOff = dstBase + mIdx * innerDim + copied;
            CopyOut(gmOut[dstOff], outData, chunk);
        }
        WaitMTE3();
        outQue.FreeTensor(outData);
    }
}

} // namespace TileKernel
#endif // TILE_H
