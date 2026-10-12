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
 * \file reflection_pad3d_grad_utils.h
 * \brief
 */
#ifndef REFLECTION_PAD3D_GRAD_UTILS_H
#define REFLECTION_PAD3D_GRAD_UTILS_H
#include <typeinfo>
#include "kernel_operator.h"

class CopyOutParam {
public:
    int64_t dstOffset;
    int64_t srcOffset;
    int64_t calH;
    int64_t calW;
    int64_t offsetWidth;
    bool isAtomicAdd;
    int64_t srcStride;
    __aicore__ inline CopyOutParam(int64_t dstOffset_tmp, int64_t srcOffset_tmp, int64_t calH_tmp, int64_t calW_tmp,
                                   int64_t offsetWidth_tmp, bool isAtomicAdd_tmp, int64_t srcStride_tmp = 0)
    {
        dstOffset = dstOffset_tmp;
        srcOffset = srcOffset_tmp;
        calH = calH_tmp;
        calW = calW_tmp;
        offsetWidth = offsetWidth_tmp;
        isAtomicAdd = isAtomicAdd_tmp;
        srcStride = srcStride_tmp;
    }
};

class CopyInParam {
public:
    int64_t dstOffset;
    int64_t srcOffset;
    int64_t calH;
    int64_t calW;
    __aicore__ inline CopyInParam(int64_t dstOffset_tmp, int64_t srcOffset_tmp, int64_t calH_tmp, int64_t calW_tmp)
    {
        dstOffset = dstOffset_tmp;
        srcOffset = srcOffset_tmp;
        calH = calH_tmp;
        calW = calW_tmp;
    }
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
};
template <typename T1, typename T2>
__aicore__ inline T1 FloorDiv(T1 a, T2 b)
{
    if (b <= 0) {
        return 0;
    }
    return (a) / (b);
};
template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};
template <typename T1, typename T2>
__aicore__ inline T1 FloorAlign(T1 a, T2 b)
{
    if (b <= 0) {
        return 0;
    }
    return (a) / b * b;
};

template <typename T>
__aicore__ inline T Mymax(T a, T b)
{
    if (a > b) {
        return a;
    }
    return b;
};

template <typename T1>
__aicore__ inline void TransDataTo5HDHelper(AscendC::LocalTensor<T1>& dstLocal, AscendC::LocalTensor<T1>& srcLocal,
                                            const int32_t calH, const int32_t calW)
{
    AscendC::TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = calH / 16;
    transDataParams.dstRepStride = (16 * sizeof(T1)) / 32;
    transDataParams.srcRepStride = (16 * calW * sizeof(T1)) / 32;
    if (transDataParams.repeatTimes == 1) {
        transDataParams.dstRepStride = 0;
        transDataParams.srcRepStride = 0;
    }
    uint64_t srcLocalList[16];
    uint64_t dstLocalList[16];
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    if constexpr (std::is_same<T1, float>::value) {
        for (int i = 0; i < calW / 8; i++) {
            for (int j = 0; j < 16; j++) {
                srcLocalList[j] = (uint64_t)(srcLocal[srcOffset + calW * j].GetPhyAddr());
            }
            for (int j = 0; j < 8; j++) {
                dstLocalList[2 * j] = (uint64_t)(dstLocal[dstOffset + calH * j].GetPhyAddr());
                dstLocalList[2 * j + 1] = (uint64_t)(dstLocal[dstOffset + calH * j + 8].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
            srcOffset += 8;
            dstOffset += 8 * calH;
        }
    } else {
        for (int i = 0; i < calW / 16; i++) {
            for (int j = 0; j < 16; j++) {
                srcLocalList[j] = (uint64_t)(srcLocal[srcOffset + calW * j].GetPhyAddr());
            }
            for (int j = 0; j < 16; j++) {
                dstLocalList[j] = (uint64_t)(dstLocal[dstOffset + calH * j].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
            srcOffset += 16;
            dstOffset += 16 * calH;
        }
    }
}

template <typename T>
__aicore__ inline void ClearOutputHelper(AscendC::GlobalTensor<T>& yGm, GM_ADDR y, uint32_t batch, uint32_t channel,
                                         uint32_t outDepth, uint32_t outHeight, uint32_t outWidth, uint32_t blockNum,
                                         uint32_t blockIdx)
{
    int64_t totaldata = batch * channel * outDepth * outHeight * outWidth;
    int64_t preLen = totaldata / blockNum;
    int64_t tailLen = totaldata % blockNum;
    int64_t curLen = preLen;
    int64_t curOffset = blockIdx * preLen;
    if (blockIdx < tailLen) {
        curLen = preLen + 1;
        curOffset = blockIdx * curLen;
    } else {
        curLen = preLen;
        curOffset = blockIdx * preLen + tailLen;
    }
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + curOffset);
    AscendC::InitGlobalMemory<T>(yGm, curLen, 0);
    AscendC::SyncAll();
}

__aicore__ inline void InitBlockHelper(uint32_t tailNC, uint32_t ncPerCore, uint32_t blockIdx, uint32_t depth,
                                       uint32_t outDepth, uint32_t dPad1, uint32_t dPad2, uint32_t& loopNC,
                                       int64_t& ncOffset, uint32_t& curDepth, uint32_t& curOutDepth)
{
    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }
    curDepth = depth;
    curOutDepth = outDepth;
    if (dPad1 == 0 && dPad2 == 0) {
        curDepth = 1;
        curOutDepth = 1;
    }
}

template <typename T>
__aicore__ inline void CopyInHelper(AscendC::LocalTensor<T>& dstLocal, AscendC::GlobalTensor<T>& srcGm,
                                    const int64_t offset, const int64_t calH, const int64_t calW,
                                    const int64_t srcStride, uint32_t perBlockCount)
{
    int64_t alignCalW = CeilAlign(calW, perBlockCount);
    int64_t alignTransCalW = CeilAlign(calW, 16);
    AscendC::DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams = {true, 0, 0, 0};
    copyParams.blockCount = calH;
    copyParams.blockLen = calW * sizeof(T);
    copyParams.srcStride = srcStride;
    copyParams.dstStride = ((alignTransCalW - alignCalW)) * sizeof(T) / 32;
    padParams.isPad = true;
    padParams.rightPadding = alignCalW - calW;
    AscendC::DataCopyPad(dstLocal, srcGm[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutHelper(AscendC::GlobalTensor<T>& dstGm, AscendC::LocalTensor<T>& srcLocal,
                                     const int64_t offset, const int64_t srcOffset, const bool isAtomicAdd,
                                     const int32_t calH, const int32_t calW, const int32_t alignTransCalW,
                                     const int32_t dstStride)
{
    AscendC::DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockCount = calH;
    copyParams.blockLen = calW * sizeof(T);
    copyParams.srcStride = (alignTransCalW - calW) * sizeof(T) / 32;
    copyParams.dstStride = dstStride;
    if (isAtomicAdd == true) {
        AscendC::SetAtomicAdd<T>();
        AscendC::DataCopyPad(dstGm[offset], srcLocal[srcOffset], copyParams);
        AscendC::SetAtomicNone();
    } else {
        AscendC::DataCopyPad(dstGm[offset], srcLocal[srcOffset], copyParams);
    }
}

template <typename T1>
__aicore__ inline void ComputeGradBasicHelper(AscendC::LocalTensor<T1>& tLocal, AscendC::LocalTensor<T1>& xLocal,
                                              size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask, size_t wPad2Mask,
                                              const int32_t calH, const int32_t calW, uint32_t hPad1, uint32_t hPad2,
                                              uint32_t wPad1, uint32_t wPad2)
{
    int64_t alignTransCalW = CeilAlign(calW, 16);
    int64_t alignTransCalH = CeilAlign(calH, 16);
    if (hPad1Mask == 0 && hPad1 > 0) {
        for (uint32_t i = 0; i < hPad1; i++) {
            auto srcLocal_1 = xLocal[i * alignTransCalW];
            auto srcLocal_2 = xLocal[(2 * hPad1 - i) * alignTransCalW];
            AscendC::Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalW);
        }
    }
    if (hPad2Mask == 0 && hPad2 > 0) {
        for (uint32_t i = 0; i < hPad2; i++) {
            auto srcLocal_1 = xLocal[(calH - 1 - i) * alignTransCalW];
            auto srcLocal_2 = xLocal[(calH - 2 * hPad2 - 1 + i) * alignTransCalW];
            AscendC::Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalW);
        }
    }
    TransDataTo5HDHelper<T1>(tLocal, xLocal, alignTransCalH, alignTransCalW);
    if (wPad1Mask == 0 && wPad1 > 0) {
        for (uint32_t i = 0; i < wPad1; i++) {
            auto srcLocal_1 = tLocal[i * alignTransCalH];
            auto srcLocal_2 = tLocal[(2 * wPad1 - i) * alignTransCalH];
            AscendC::Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalH);
        }
    }
    if (wPad2Mask == 0 && wPad2 > 0) {
        for (uint32_t i = 0; i < wPad2; i++) {
            auto srcLocal_1 = tLocal[(calW - 1 - i) * alignTransCalH];
            auto srcLocal_2 = tLocal[(calW - 2 * wPad2 - 1 + i) * alignTransCalH];
            AscendC::Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalH);
        }
    }
    // 平移
    if (wPad1Mask == 0 && wPad1 > 0) {
        for (uint32_t i = 0; i < calW - wPad1; i++) {
            auto srcLocal_1 = tLocal[i * alignTransCalH];
            auto srcLocal_2 = tLocal[(i + wPad1) * alignTransCalH];
            AscendC::Muls(srcLocal_1, srcLocal_2, (T1)1.0, alignTransCalH);
        }
    }
}

#endif // REFLECTION_PAD3D_GRAD_UTILS_H
