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
 * \file sinkhorn_update.inc
 * \brief
 */
#ifndef SINKHORN_KERNEL_UPDATE_INC_
#define SINKHORN_KERNEL_UPDATE_INC_

namespace AscendC {
    template<typename T, typename IT>
    __aicore__ inline T KernelSinkhorn<T, IT>::CpuCascadeRowSum(LocalTensor<T> srcLocal, uint32_t size)
    {
        if (size < CPU_SUM_GROUP_SIZE) {
            T partial[CPU_SUM_ILP] = {0, 0, 0, 0};
            uint32_t sizeIlp = size / CPU_SUM_ILP;
            for (uint32_t i = 0; i < sizeIlp; ++i) {
                for (uint32_t k = 0; k < CPU_SUM_ILP; ++k) {
                    partial[k] += srcLocal.GetValue(i * CPU_SUM_ILP + k);
                }
            }
            for (uint32_t i = sizeIlp * CPU_SUM_ILP; i < size; ++i) {
                partial[0] += srcLocal.GetValue(i);
            }
            for (uint32_t k = 1; k < CPU_SUM_ILP; ++k) {
                partial[0] += partial[k];
            }
            return partial[0];
        }

        LocalTensor<T> acc = cpuSumScratch.Get<T>();
        Duplicate(acc, static_cast<T>(0), CPU_SUM_SCRATCH_SIZE);
        PipeBarrier<PIPE_V>();

        uint32_t vecSize = size / CPU_SUM_GROUP_SIZE;
        uint32_t sizeIlp = vecSize / CPU_SUM_ILP;
        uint32_t fullCascadeSize = sizeIlp / CPU_SUM_GROUP_SIZE * CPU_SUM_GROUP_SIZE;

        for (uint32_t i = 0; i < fullCascadeSize; ++i) {
            for (uint32_t k = 0; k < CPU_SUM_ILP; ++k) {
                uint32_t dst = k * CPU_SUM_GROUP_SIZE;
                uint32_t src = (i * CPU_SUM_ILP + k) * CPU_SUM_GROUP_SIZE;
                Add(acc[dst], acc[dst], srcLocal[src], CPU_SUM_GROUP_SIZE);
            }
            PipeBarrier<PIPE_V>();

            uint32_t count = i + 1;
            if ((count & (CPU_SUM_GROUP_SIZE - 1)) == 0) {
                for (uint32_t level = 1; level < CPU_SUM_LEVELS; ++level) {
                    for (uint32_t k = 0; k < CPU_SUM_ILP; ++k) {
                        uint32_t lower = ((level - 1) * CPU_SUM_ILP + k) * CPU_SUM_GROUP_SIZE;
                        uint32_t upper = (level * CPU_SUM_ILP + k) * CPU_SUM_GROUP_SIZE;
                        Add(acc[upper], acc[upper], acc[lower], CPU_SUM_GROUP_SIZE);
                        Duplicate(acc[lower], static_cast<T>(0), CPU_SUM_GROUP_SIZE);
                    }
                    PipeBarrier<PIPE_V>();
                    uint32_t mask = (CPU_SUM_GROUP_SIZE - 1) << (level * 4);
                    if ((count & mask) != 0) {
                        break;
                    }
                }
            }
        }

        for (uint32_t i = fullCascadeSize; i < sizeIlp; ++i) {
            for (uint32_t k = 0; k < CPU_SUM_ILP; ++k) {
                uint32_t dst = k * CPU_SUM_GROUP_SIZE;
                uint32_t src = (i * CPU_SUM_ILP + k) * CPU_SUM_GROUP_SIZE;
                Add(acc[dst], acc[dst], srcLocal[src], CPU_SUM_GROUP_SIZE);
            }
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t i = sizeIlp * CPU_SUM_ILP; i < vecSize; ++i) {
            Add(acc, acc, srcLocal[i * CPU_SUM_GROUP_SIZE], CPU_SUM_GROUP_SIZE);
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t k = 0; k < CPU_SUM_ILP; ++k) {
            uint32_t base = k * CPU_SUM_GROUP_SIZE;
            for (uint32_t level = 1; level < CPU_SUM_LEVELS; ++level) {
                uint32_t upper = (level * CPU_SUM_ILP + k) * CPU_SUM_GROUP_SIZE;
                Add(acc[base], acc[base], acc[upper], CPU_SUM_GROUP_SIZE);
                PipeBarrier<PIPE_V>();
            }
        }
        for (uint32_t k = 1; k < CPU_SUM_ILP; ++k) {
            Add(acc, acc, acc[k * CPU_SUM_GROUP_SIZE], CPU_SUM_GROUP_SIZE);
            PipeBarrier<PIPE_V>();
        }

        T sum = 0;
        for (uint32_t i = vecSize * CPU_SUM_GROUP_SIZE; i < size; ++i) {
            sum += srcLocal.GetValue(i);
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        for (uint32_t i = 0; i < CPU_SUM_GROUP_SIZE; ++i) {
            sum += acc.GetValue(i);
        }
        return sum;
    }

    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInFromP(uint16_t row, const GlobalTensor<_IT> &pG)
    {
        // 搬入cost
        uint16_t blockLen = totalCol * sizeof(T);
        LocalTensor<T> costLocal = costInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{(uint16_t)row, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(costLocal, pG, copyParams, padParams);
        costInQueue.EnQue(costLocal);
    }

    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SaveP(uint16_t row, const GlobalTensor<_IT> &pG, const LocalTensor<T> &localTensor)
    {
        uint16_t blockLen = totalCol * sizeof(T);
        DataCopyExtParams copyParams{(uint16_t)row, blockLen, 0, 0, 0};
        DataCopyPad(pG, localTensor, copyParams);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD0(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d1InLocal)
    {
        // costOutQueue = d1 * cost: 逐行计算 Mul
        {
            LocalTensor<T> costDstLocal = costOutQueue.AllocTensor<T>();
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                Mul(costDstLocal[rowIdx], costSrcLocal[rowIdx], d1InLocal, totalCol);
            }
            costOutQueue.EnQue(costDstLocal);
        }

        // d0OutLocal = sum
        {
            LocalTensor<T> d0OutLocal = d0OutQueue.AllocTensor<T>();
            LocalTensor<T> d0OutLocal2 = d1OutQueue2.AllocTensor<T>();
            LocalTensor<T> costDstLocal = costOutQueue.DeQue<T>();

            event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId_V_S);
            WaitFlag<HardEvent::V_S>(eventId_V_S);
            if (totalCol < CPU_SUM_GROUP_SIZE) {
                if constexpr (!std::is_same<T, IT>::value && !std::is_same<IT, bfloat16_t>::value) {
                    event_t eventId_MTE2_S =
                        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                    SetFlag<HardEvent::MTE2_S>(eventId_MTE2_S);
                    WaitFlag<HardEvent::MTE2_S>(eventId_MTE2_S);
                    for (int r = 0; r < row; r++) {
                        uint32_t rowIdx = r * rowLengthAligned;
                        T sum = static_cast<T>(0);
                        for (int c = 0; c < totalCol; c++) {
                            sum += costSrcLocal.GetValue(rowIdx + c) * d1InLocal.GetValue(c);
                        }
                        d0OutLocal.SetValue(r, sum);
                    }
                } else {
                    LocalTensor<T> reduceScratch = cpuSumScratch.Get<T>();
                    for (int r = 0; r < row; r++) {
                        uint32_t rowIdx = r * rowLengthAligned;
                        ReduceSum(d0OutLocal2, costDstLocal[rowIdx], reduceScratch, totalCol);
                        SetFlag<HardEvent::V_S>(eventId_V_S);
                        WaitFlag<HardEvent::V_S>(eventId_V_S);
                        d0OutLocal.SetValue(r, d0OutLocal2.GetValue(0));
                    }
                }
            } else {
                for (int r = 0; r < row; r++) {
                    uint32_t rowIdx = r * rowLengthAligned;
                    T sum = CpuCascadeRowSum(costDstLocal[rowIdx], totalCol);
                    d0OutLocal.SetValue(r, sum);
                }
            }
            event_t eventId_S_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventId_S_V);
            WaitFlag<HardEvent::S_V>(eventId_S_V);
            d0OutQueue.EnQue(d0OutLocal);
            d1OutQueue2.FreeTensor(d0OutLocal2);
            costOutQueue.FreeTensor(costDstLocal);
        }

        // + eps
        {
            LocalTensor<T> d0OutLocal2 = d0OutQueue2.AllocTensor<T>();
            LocalTensor<T> d0OutLocal3 = d0OutQueue3.AllocTensor<T>();
            LocalTensor<T> d0OutLocal = d0OutQueue.DeQue<T>();
            DUMP_LT_3(d0OutLocal, row, "d0[sum]: ");
            Adds(d0OutLocal2, d0OutLocal, (T)eps, row);
            DUMP_LT_3(d0OutLocal2, row, "d0[eps]: ");
            PipeBarrier<PIPE_V>();
            Duplicate(d0OutLocal3, static_cast<T>(1.0f), row);
            PipeBarrier<PIPE_V>();
            Div(d0OutLocal, d0OutLocal3, d0OutLocal2, row);
            PipeBarrier<PIPE_V>();
            Mul(d0OutLocal3, d0OutLocal2, d0OutLocal, row);
            PipeBarrier<PIPE_V>();
            Muls(d0OutLocal3, d0OutLocal3, static_cast<T>(-1.0f), row);
            PipeBarrier<PIPE_V>();
            Adds(d0OutLocal3, d0OutLocal3, static_cast<T>(2.0f), row);
            PipeBarrier<PIPE_V>();
            Mul(d0OutLocal, d0OutLocal, d0OutLocal3, row);
            PipeBarrier<PIPE_V>();
            Muls(d0OutLocal, d0OutLocal, static_cast<T>(1.0f / totalRow), row);
            DUMP_LT_3(d0OutLocal, row, "d0[div]: ");
            d0OutQueue.EnQue(d0OutLocal);
            d0OutQueue3.FreeTensor(d0OutLocal3);
            d0OutQueue2.FreeTensor(d0OutLocal2);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD1CpuPartials()
    {
        uint32_t groupCount = (totalRow + CPU_SUM_GROUP_SIZE - 1) / CPU_SUM_GROUP_SIZE;
        DataCopyExtParams rowCopy{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> rowPad{false, 0, 0, 0};

        for (uint32_t group = blockIdx; group < groupCount; group += numBlocks) {
            uint32_t firstRow = group * CPU_SUM_GROUP_SIZE;
            uint32_t rows = totalRow - firstRow;
            if (rows > CPU_SUM_GROUP_SIZE) {
                rows = CPU_SUM_GROUP_SIZE;
            }

            LocalTensor<T> d0Local = d0InQueue.AllocTensor<T>();
            DataCopyExtParams d0Copy{1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> d0Pad{false, 0, 0, 0};
            DataCopyPad(d0Local, d0GlobalInWS[firstRow], d0Copy, d0Pad);
            d0InQueue.EnQue(d0Local);
            d0Local = d0InQueue.DeQue<T>();

            LocalTensor<T> sumLocal = d1OutQueue.AllocTensor<T>();
            Duplicate(sumLocal, static_cast<T>(0), totalCol);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < rows; ++r) {
                LocalTensor<T> expLocal = costInQueue.AllocTensor<T>();
                DataCopyPad(expLocal, pFloatAllGlobal[(firstRow + r) * totalCol], rowCopy, rowPad);
                costInQueue.EnQue(expLocal);
                expLocal = costInQueue.DeQue<T>();

                LocalTensor<T> productLocal = costOutQueue.AllocTensor<T>();
                Muls(productLocal, expLocal, d0Local.GetValue(r), totalCol);
                PipeBarrier<PIPE_V>();
                Add(sumLocal, sumLocal, productLocal, totalCol);
                PipeBarrier<PIPE_V>();

                costOutQueue.FreeTensor(productLocal);
                costInQueue.FreeTensor(expLocal);
            }

            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            DataCopyPad(d1BlockInWS[group * totalCol], sumLocal, rowCopy);
            d1OutQueue.FreeTensor(sumLocal);
            d0InQueue.FreeTensor(d0Local);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ReduceD1CpuPartials()
    {
        if (blockIdx != 0) {
            return;
        }

        uint32_t fullGroups = totalRow / CPU_SUM_GROUP_SIZE;
        bool hasTail = (totalRow % CPU_SUM_GROUP_SIZE) != 0;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> level0 = d1OutQueue.AllocTensor<T>();
        LocalTensor<T> level1 = d1OutQueue2.AllocTensor<T>();
        LocalTensor<T> level2 = d1OutQueue3.AllocTensor<T>();
        LocalTensor<T> level3 = d1InQueue.AllocTensor<T>();
        Duplicate(level0, static_cast<T>(0), totalCol);
        Duplicate(level1, static_cast<T>(0), totalCol);
        Duplicate(level2, static_cast<T>(0), totalCol);
        Duplicate(level3, static_cast<T>(0), totalCol);
        PipeBarrier<PIPE_V>();

        for (uint32_t group = 0; group < fullGroups; ++group) {
            LocalTensor<T> partial = d1OldInQueue.AllocTensor<T>();
            DataCopyPad(partial, d1BlockInWS[group * totalCol], copyParams, padParams);
            d1OldInQueue.EnQue(partial);
            partial = d1OldInQueue.DeQue<T>();
            Add(level1, level1, partial, totalCol);
            PipeBarrier<PIPE_V>();
            d1OldInQueue.FreeTensor(partial);

            uint32_t count = group + 1;
            if ((count & 15) == 0) {
                Add(level2, level2, level1, totalCol);
                Duplicate(level1, static_cast<T>(0), totalCol);
                PipeBarrier<PIPE_V>();
                if ((count & 255) == 0) {
                    Add(level3, level3, level2, totalCol);
                    Duplicate(level2, static_cast<T>(0), totalCol);
                    PipeBarrier<PIPE_V>();
                }
            }
        }

        if (hasTail) {
            DataCopyPad(level0, d1BlockInWS[fullGroups * totalCol], copyParams, padParams);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }
        Add(level0, level0, level1, totalCol);
        PipeBarrier<PIPE_V>();
        Add(level0, level0, level2, totalCol);
        PipeBarrier<PIPE_V>();
        Add(level0, level0, level3, totalCol);
        d1OutQueue.EnQue(level0);

        d1InQueue.FreeTensor(level3);
        d1OutQueue3.FreeTensor(level2);
        d1OutQueue2.FreeTensor(level1);
    }

    // 计算每个Tile的d1   torch.sum(d0.unsqueeze(1) * cost, 1)
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD1(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d0OutLocal)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        SetAtomicAdd<T>();
        event_t eventId_MTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        for (int r = 0; r < row; r++) {
            uint32_t rowIdx = r * rowLengthAligned;
            T d0 = d0OutLocal.GetValue(r);
            WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
            Muls(d1OutLocal, costSrcLocal[rowIdx], d0, totalCol);
            event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
            DataCopyPad(d1BlockInWS, d1OutLocal, copyParams);
            SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        }
        WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        SetAtomicNone();
        d1OutQueue.FreeTensor(d1OutLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInD1BlockInWS(DataCopyExtParams copyParams, DataCopyPadExtParams<T> padParams)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
        DataCopyPad(d1OutLocal, d1BlockInWS, copyParams, padParams);
        d1OutQueue.EnQue(d1OutLocal);
        DUMP_LT_2(d1BlockInWS, totalCol, " d1[sm1]: ");
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SumD1Block(DataCopyExtParams copyParams)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
        DUMP_LT_2(d1OutLocal, totalCol, " d1[sm2]: ");
        SetAtomicAdd<T>();
        DataCopyPad(d1GlobalInWSNew, d1OutLocal, copyParams);
        DataCacheClean(d1GlobalInWSNew);
        SetAtomicNone();
        d1OutQueue.FreeTensor(d1OutLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::UpdateD0()
    {
        // d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)

        DataCacheClean(d1GlobalInWS);
        DataCacheClean(d1GlobalInWSNew);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // d1InQueue = d1
        {
            LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
            DataCopyPad(d1InLocal, d1GlobalInWS, copyParams, padParams);
            d1InQueue.EnQue(d1InLocal);
        }

        LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();

        for (int t = 0; t < tileNum; t++) {
            uint32_t tileIdx = t * this->tileLength;
            uint32_t length = this->tileLength;
            uint32_t row = tileRow;
            if (t == this->tileNum - 1) {
                length = this->lastTileLength;
                row = lastTileRow;
            }

            // 搬入cost
            CopyInFromP<T>(row, pFloatGlobal[tileIdx]);

            // 计算d0
            LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
            ComputeD0(row, costSrcLocal, d1InLocal);

            LocalTensor<T> d0OutLocal = d0OutQueue.DeQue<T>();

            // copy to workspace
            {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(row * sizeof(T)), 0, 0, 0};
                DataCopyPad(d0BlockInWS[tileRow * t], d0OutLocal, copyParams);
            }
            DUMP_LT_2(d1BlockInWS, totalCol, "d1[s01]: ");

            d0OutQueue.FreeTensor(d0OutLocal);
            costInQueue.FreeTensor(costSrcLocal);
        }

        d1InQueue.FreeTensor(d1InLocal);

        SyncAll();
        ComputeD1CpuPartials();
        SyncAll();
        ReduceD1CpuPartials();
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(headerInWS);
        DUMP_LT_0_2(d0BlockInWS, blockRow, "d0 : ");
#ifdef MULTI_CORE_SUM_FOR_D1
        DUMP_LT_0_2(d1GlobalInWSNew, totalCol, "d1[sum] : ");
#endif
    }

#include "sinkhorn_update_d1.inc"
} // namespace AscendC

#endif // SINKHORN_KERNEL_UPDATE_INC_
