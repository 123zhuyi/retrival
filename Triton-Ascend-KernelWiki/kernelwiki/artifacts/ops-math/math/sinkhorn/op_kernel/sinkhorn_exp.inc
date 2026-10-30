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
 * \file sinkhorn_exp.inc
 * \brief
 */
#ifndef SINKHORN_KERNEL_EXP_INC_
#define SINKHORN_KERNEL_EXP_INC_

namespace AscendC {
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ExpCost()
    {
        for (int i = 0; i < tileNum; i++) {
            uint32_t ind = i * tileLength;
            uint32_t length = tileLength;
            if (i == tileNum - 1) {
                length = lastTileLength;
            }
            CopyInForExp(ind, length);
            ComputeForExp(length);
            CopyOutForExp<IT>(ind, length);
        }
    }

    // 通用float/half
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInForExp(uint32_t ind, uint32_t length)
    {
        LocalTensor<T> costLocal = costInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(costLocal, costGlobal[ind], copyParams, padParams);
        costInQueue.EnQue(costLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeForExp(uint32_t length)
    {
        LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
        LocalTensor<T> costDstLocal = costOutQueue.AllocTensor<T>();
        DUMP_LT_2(costSrcLocal, length, "cost: ");
        Exp(costDstLocal, costSrcLocal, length);
        costOutQueue.EnQue<T>(costDstLocal);
        costInQueue.FreeTensor(costSrcLocal);
    }

    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyOutForExp(uint32_t ind, uint32_t length)
    {
        LocalTensor<T> costLocal = costOutQueue.DeQue<T>();
        DUMP_LT_2(costLocal, length, " exp: ");
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPad(pGlobal[ind], costLocal, copyParams);
        costOutQueue.FreeTensor(costLocal);
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::CopyInForExp(uint32_t ind, uint32_t length)
    {
        // 从costGlobal复制到tmpLocal
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmpLocal, costGlobal[ind], copyParams, padParams);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> float
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        LocalTensor<float> costLocal = costInQueue.AllocTensor<float>();
        Cast(costLocal, tmpLocal, RoundMode::CAST_NONE, length);
        costInQueue.EnQue(costLocal);
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = half
    template<>
    __aicore__ inline void KernelSinkhorn<float, half>::CopyInForExp(uint32_t ind, uint32_t length)
    {
        // 从costGlobal复制到tmpLocal
        LocalTensor<half> tmpLocal = costOutQueue.AllocTensor<half>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(half)), 0, 0, 0};
        DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
        DataCopyPad(tmpLocal, costGlobal[ind], copyParams, padParams);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> float
        tmpLocal = costOutQueue.DeQue<half>();
        LocalTensor<float> costLocal = costInQueue.AllocTensor<float>();
        Cast(costLocal, tmpLocal, RoundMode::CAST_NONE, length);
        costInQueue.EnQue(costLocal);
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::CopyOutForExp<bfloat16_t>(uint32_t ind, uint32_t length)
    {
        // float ==> bf16
        LocalTensor<float> costLocal = costOutQueue.DeQue<float>();
        LocalTensor<bfloat16_t> tmpLocal = costInQueue.AllocTensor<bfloat16_t>();
        Cast(tmpLocal, costLocal, RoundMode::CAST_TRUNC, length);
        costInQueue.EnQue(tmpLocal);

        // bf16 ==> pGlobal
        tmpLocal = costInQueue.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(pGlobal[ind], tmpLocal, copyParams);
        costInQueue.FreeTensor(tmpLocal);
        costOutQueue.FreeTensor(costLocal);
    }

    // 特殊处理 T = float, IT = half
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, half>::CopyOutForExp<half>(uint32_t ind, uint32_t length)
    {
        // float ==> bf16
        LocalTensor<float> costLocal = costOutQueue.DeQue<float>();
        LocalTensor<half> tmpLocal = costInQueue.AllocTensor<half>();
        Cast(tmpLocal, costLocal, RoundMode::CAST_TRUNC, length);
        costInQueue.EnQue(tmpLocal);

        // bf16 ==> pGlobal
        tmpLocal = costInQueue.DeQue<half>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(half)), 0, 0, 0};
        DataCopyPad(pGlobal[ind], tmpLocal, copyParams);
        costInQueue.FreeTensor(tmpLocal);
        costOutQueue.FreeTensor(costLocal);
    }
} // namespace AscendC

#endif // SINKHORN_KERNEL_EXP_INC_
