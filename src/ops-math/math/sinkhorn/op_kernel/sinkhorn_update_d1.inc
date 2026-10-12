/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SINKHORN_KERNEL_UPDATE_D1_INC_
#define SINKHORN_KERNEL_UPDATE_D1_INC_

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SumD1(int block)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // copy from core level
        {
            LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
            DataCopyPad(d1InLocal, d1BlockInWS[block * totalCol], copyParams, padParams);
            d1InQueue.EnQue(d1InLocal);
        }
        {
            LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
            LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();
            DUMP_LT_2(d1InLocal, totalCol, " d1[   ]: ");
            Add(d1OutLocal, d1InLocal, d1OutLocal, totalCol);
            d1OutQueue.EnQue(d1OutLocal);
            d1InQueue.FreeTensor(d1InLocal);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::UpdateD1()
    {
        if (blockIdx == 0) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

            // d1OutQueue2 = new d1 + eps
            {
                LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
                DUMP_LT_2(d1OutLocal, totalCol, " d1[sum]: ");
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.AllocTensor<T>();
                Adds(d1OutLocal2, d1OutLocal, (T)eps, totalCol);
                d1OutQueue2.EnQue(d1OutLocal2);
                d1OutQueue.FreeTensor(d1OutLocal);
            }

            // d1OutQueue3 = 1.0
            {
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                Duplicate(d1OutLocal3, static_cast<T>(1.0f), totalCol);
                d1OutQueue3.EnQue(d1OutLocal3);
            }

            // d1InQueue = (1.0 / d1 new) / totalCol
            {
                LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.DeQue<T>();
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.DeQue<T>();

                Div(d1InLocal, d1OutLocal3, d1OutLocal2, totalCol);
                PipeBarrier<PIPE_V>();
                Mul(d1OutLocal3, d1OutLocal2, d1InLocal, totalCol);
                PipeBarrier<PIPE_V>();
                Muls(d1OutLocal3, d1OutLocal3, static_cast<T>(-1.0f), totalCol);
                PipeBarrier<PIPE_V>();
                Adds(d1OutLocal3, d1OutLocal3, static_cast<T>(2.0f), totalCol);
                PipeBarrier<PIPE_V>();
                Mul(d1InLocal, d1InLocal, d1OutLocal3, totalCol);
                PipeBarrier<PIPE_V>();
                Muls(d1InLocal, d1InLocal, static_cast<T>(1.0f / totalCol), totalCol);
                d1InQueue.EnQue(d1InLocal);
                d1OutQueue2.FreeTensor(d1OutLocal2);
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            // 取出新 d1
            LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();

            // 旧 d1 搬入专用 VECIN 队列 d1OldInQueue：EnQue/DeQue 自带 MTE2->V 同步，
            // 不再复用 d1OutQueue2 跨 V/MTE2，规避手动 MTE2_V 与活跃 VECIN 队列抢 event id（报错4/5/6/7/8）
            LocalTensor<T> d1OldLocal = d1OldInQueue.AllocTensor<T>();
            DataCopyPad(d1OldLocal, d1GlobalInWS, copyParams, padParams);
            d1OldInQueue.EnQue(d1OldLocal);

            // 写新值（MTE3）：同一 d1GlobalInWS 区，旧 d1 的 MTE2 读须先完成，补 MTE2->MTE3 同步（报错9）
            {
                event_t eventIdMte2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2Mte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2Mte3);
                DataCopyPad(d1GlobalInWS, d1InLocal, copyParams);
                DataCacheClean(d1GlobalInWS);
            }

            // 清空新值
            {
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                Duplicate(d1OutLocal3, static_cast<T>(0.0f), totalCol);
                event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
                DataCopyPad(d1GlobalInWSNew, d1OutLocal3, copyParams);
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            // 收敛判断：mean(|old_d1 - new_d1|)
            {
                DataCacheClean(d1GlobalInWSNew);
                LocalTensor<T> d1OldLocalIn = d1OldInQueue.DeQue<T>();
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.AllocTensor<T>();

                // d1_old - d1
                Sub(d1OutLocal3, d1OldLocalIn, d1InLocal, totalCol);
                DUMP_LT_3(d1OutLocal3, totalCol, " d1[sub]: ");
                PipeBarrier<PIPE_V>();
                Abs(d1OutLocal, d1OutLocal3, totalCol);
                DUMP_LT_3(d1OutLocal, totalCol, " d1[abs]: ");
                PipeBarrier<PIPE_V>();
                ReduceSum(d1OutLocal2, d1OutLocal, d1OutLocal3, totalCol);
                event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId_V_S);
                WaitFlag<HardEvent::V_S>(eventId_V_S);
                float tolerance = (float)(d1OutLocal2.GetValue(0)) / totalCol;
                OP_LOGD_0_1("tol[%d]: " FLOAT_FMT, loopCount, tolerance);
                if (!(tolerance > tol)) {
                    // 退出循环
                    SetLoopFlag(0);
                }
                d1OutQueue.FreeTensor(d1OutLocal);
                d1OutQueue3.FreeTensor(d1OutLocal3);
                d1OutQueue2.FreeTensor(d1OutLocal2);
                d1OldInQueue.FreeTensor(d1OldLocalIn);
            }

            d1InQueue.FreeTensor(d1InLocal);
        }

        SyncAll();
        DataCacheClean(d1GlobalInWS);
        DataCacheClean(d1GlobalInWSNew);
        DUMP_LT_0_2(d1GlobalInWS, totalCol, " d1[gl ]: ");
        DUMP_LT_0_2(d1GlobalInWSNew, totalCol, " d1[gln]: ");
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::SaveP<bfloat16_t>(uint16_t row, const GlobalTensor<bfloat16_t> &pG, const LocalTensor<float> &localTensor)
    {
        // float ==> bf16
        uint32_t blockLen = totalCol * sizeof(bfloat16_t);
        uint32_t outputRowAligned = (blockLen + 31) / 32 * 32 / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        for (int r = 0; r < row; r++) {
            Cast(tmpLocal[r * outputRowAligned], localTensor[r * rowLengthAligned], RoundMode::CAST_RINT, totalCol);
        }
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> GM
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        for (int r = 0; r < row; r++) {
            DataCopyPad(pG[r * totalCol], tmpLocal[r * outputRowAligned], copyParams);
        }
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = half
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, half>::SaveP<half>(uint16_t row, const GlobalTensor<half> &pG, const LocalTensor<float> &localTensor)
    {
        // float ==> fp16
        uint32_t blockLen = totalCol * sizeof(half);
        uint32_t outputRowAligned = (blockLen + 31) / 32 * 32 / sizeof(half);
        LocalTensor<half> tmpLocal = costOutQueue.AllocTensor<half>();
        for (int r = 0; r < row; r++) {
#if __NPU_ARCH__ == 3510
            asc_vf_call<SinkhornFloatToHalfVF>(
                (__ubuf__ half*)tmpLocal[r * outputRowAligned].GetPhyAddr(),
                (__ubuf__ float*)localTensor[r * rowLengthAligned].GetPhyAddr(), totalCol);
#else
            Cast(tmpLocal[r * outputRowAligned], localTensor[r * rowLengthAligned], RoundMode::CAST_RINT, totalCol);
#endif
        }
        costOutQueue.EnQue(tmpLocal);

        // fp16 ==> GM
        tmpLocal = costOutQueue.DeQue<half>();
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        for (int r = 0; r < row; r++) {
            DataCopyPad(pG[r * totalCol], tmpLocal[r * outputRowAligned], copyParams);
        }
        costOutQueue.FreeTensor(tmpLocal);
    }

#endif // SINKHORN_KERNEL_UPDATE_D1_INC_
