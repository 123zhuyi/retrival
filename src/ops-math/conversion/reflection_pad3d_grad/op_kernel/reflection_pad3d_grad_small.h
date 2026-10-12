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
 * \file reflection_pad3d_grad_small.h
 * \brief
 */
#ifndef REFLECTION_PAD3D_GRAD_SMALL_H
#define REFLECTION_PAD3D_GRAD_SMALL_H
#include "reflection_pad3d_grad_init.h"

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::SmallProcess()
{
    int64_t gmXOffset = 0;
    int64_t gmYOffset = 0;
    for (size_t loop = 0; loop < loopNC; loop++) {
        for (size_t i = 0; i < curDepth; i++) {
            size_t cur_D = GetCurD(i);
            bool isAtomicAdd = true;
            // top
            gmXOffset = (loop * curDepth * height * width + i * height * width);
            gmYOffset = (loop * curOutDepth * outHeight * outWidth + cur_D * outHeight * outWidth);
            CopyInSmall(gmXOffset, height, width, 0);
            ComputeSmall(0, 0, 0, 0, height, width);
            CopyOutSmall(gmYOffset, hPad1 * alignWidth, isAtomicAdd, outHeight, outWidth, alignWidth, 0);
            PipeBarrier<PIPE_MTE3>();
            ;
        }
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyInSmall(const int64_t offset, const int64_t calH, const int64_t calW,
                                                           const int64_t srcStride)
{
    LocalTensor<T> dstLocal = inQueueX.AllocTensor<T>();
    CopyInHelper<T>(dstLocal, xGm, offset, calH, calW, srcStride, perBlockCount);
    inQueueX.EnQue(dstLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeSmall(size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask,
                                                            size_t wPad2Mask, const int32_t calH, const int32_t calW)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    int32_t alignHeight = CeilAlign(calH, 16);
    int32_t alignWidth = CeilAlign(calW, 16);
    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        LocalTensor<float> tLocal = transposeBuf.Get<float>();
        LocalTensor<float> float32Tensor = float32Buf.Get<float>();
        int32_t totalData = alignHeight * alignWidth;
        Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
        ComputeSmallBasic<float>(tLocal, float32Tensor, hPad1Mask, hPad2Mask, wPad1Mask, wPad2Mask, calH, calW);
        TransoseSmall<float>(float32Tensor, tLocal, alignWidth, alignHeight);
        Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
    } else {
        LocalTensor<T> tLocal = transposeBuf.Get<T>();
        ComputeSmallBasic<T>(tLocal, xLocal, hPad1Mask, hPad2Mask, wPad1Mask, wPad2Mask, calH, calW);
        TransoseSmall<T>(yLocal, tLocal, alignWidth, alignHeight);
    }
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyOutSmall(const int64_t offset, const int64_t srcOffset,
                                                            const bool isAtomicAdd, const int32_t calH,
                                                            const int32_t calW, const int32_t alignTransCalW,
                                                            const int32_t dstStride)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    CopyOutHelper<T>(yGm, yLocal, offset, srcOffset, isAtomicAdd, calH, calW, alignTransCalW, dstStride);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
template <typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::TransoseSmall(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal,
                                                             const int32_t calH, const int32_t calW)
{
    TransDataTo5HDHelper<T1>(dstLocal, srcLocal, calH, calW);
}

template <typename T>
template <typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeSmallBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal,
                                                                 size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask,
                                                                 size_t wPad2Mask, const int32_t calH,
                                                                 const int32_t calW)
{
    ComputeGradBasicHelper<T1>(tLocal, xLocal, hPad1Mask, hPad2Mask, wPad1Mask, wPad2Mask, calH, calW, hPad1, hPad2,
                               wPad1, wPad2);
}

#endif // REFLECTION_PAD3D_GRAD_SMALL_H
