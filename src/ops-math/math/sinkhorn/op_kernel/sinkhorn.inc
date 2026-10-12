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
 * \file sinkhorn.inc
 * \brief
 */
#ifndef SINKHORN_KERNEL_INC_
#define SINKHORN_KERNEL_INC_

namespace AscendC {
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::Init(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, const SinkhornTilingData *tilingData)
    {
        // 全局信息
        this->numBlocks = GetBlockNum();
        this->blockIdx = GetBlockIdx();

        OP_LOGD_0_1("tol[in]:  " FLOAT_FMT, tol);

        uint64_t formerNum = tilingData->formerNum;
        uint64_t formerLength = tilingData->formerLength;
        uint64_t tailLength = tilingData->tailLength;

        if (blockIdx < formerNum) {
            tileNum = tilingData->formerTileNum;
            lastTileRow = tilingData->formerLastTileRow;
            lastTileLength = tilingData->formerLastTileLength;
        } else {
            tileNum = tilingData->tailTileNum;
            lastTileRow = tilingData->tailLastTileRow;
            lastTileLength = tilingData->tailLastTileLength;
        }

        this->tileRow = tilingData->tileRow;
        this->tileLength = tilingData->tileLength;

        this->totalRow = tilingData->totalRow;
        this->totalCol = tilingData->totalCol;
        this->totalColAligned = tilingData->totalColAligned;

        this->tol = tilingData->tol;

        if (blockIdx < formerNum) {
            // former
            costGlobal.SetGlobalBuffer((__gm__ IT*)cost + blockIdx * formerLength, formerLength);
            pGlobal.SetGlobalBuffer((__gm__ IT*)p + blockIdx * formerLength, formerLength);
        } else {
            // tail
            costGlobal.SetGlobalBuffer((__gm__ IT*)cost + formerNum * formerLength
                + (blockIdx - formerNum) * tailLength, tailLength);
            pGlobal.SetGlobalBuffer((__gm__ IT*)p + formerNum * formerLength
                + (blockIdx - formerNum) * tailLength, tailLength);
        }

        InitWS(workspace, (blockIdx < formerNum), formerNum, tilingData->formerRow, tilingData->tailRow);
        InitUB();
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::InitWS(GM_ADDR workspace, bool isFormer, uint64_t formerNum, uint64_t formerRow, uint64_t tailRow) {
        // workspace
        GM_ADDR newBegin = workspace;
        headerInWS.SetGlobalBuffer((__gm__ uint64_t*)newBegin, HEADER_SIZE_IN_INT64);
        newBegin += sizeof(uint64_t) * HEADER_SIZE_IN_INT64;

        d0GlobalInWS.SetGlobalBuffer((__gm__ T*)newBegin, this->totalRow);
        if (isFormer) {
            // former
            blockRow = formerRow;
            d0BlockInWS.SetGlobalBuffer((__gm__ T*)newBegin
                + blockIdx * formerRow, formerRow);
        } else {
            // tail
            blockRow = tailRow;
            d0BlockInWS.SetGlobalBuffer((__gm__ T*)newBegin
                + formerNum * formerRow + (blockIdx - formerNum) * tailRow, tailRow);
        }
        newBegin += totalRow * sizeof(T);

        uint64_t d1GroupCount = (totalRow + CPU_SUM_GROUP_SIZE - 1) / CPU_SUM_GROUP_SIZE;
        d1BlockInWS.SetGlobalBuffer((__gm__ T*)newBegin, d1GroupCount * totalCol);
        newBegin += d1GroupCount * totalCol * sizeof(T);

        d1GlobalInWS.SetGlobalBuffer((__gm__ T*)newBegin, totalCol);
        newBegin += totalCol * sizeof(T);

        d1GlobalInWSNew.SetGlobalBuffer((__gm__ T*)newBegin, totalCol);
        newBegin += totalCol * sizeof(T);

        uint64_t blockOffset;
        uint64_t blockLength;
        if (isFormer) {
            blockOffset = blockIdx * formerRow * totalCol;
            blockLength = formerRow * totalCol;
        } else {
            blockOffset = (formerNum * formerRow + (blockIdx - formerNum) * tailRow) * totalCol;
            blockLength = tailRow * totalCol;
        }
        pFloatAllGlobal.SetGlobalBuffer((__gm__ T*)newBegin, totalRow * totalCol);
        pFloatGlobal.SetGlobalBuffer((__gm__ T*)newBegin + blockOffset, blockLength);
        OP_LOGD_0_3("workspace: %d", newBegin - workspace);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::InitUB()
    {
        pipe.InitBuffer(costInQueue, COST_BUFFER_NUM, tileRow * totalColAligned * sizeof(T));
        pipe.InitBuffer(costOutQueue, COST_BUFFER_NUM, tileRow * totalColAligned * sizeof(T));

        pipe.InitBuffer(d0InQueue, 1, tileRow * sizeof(T));
        pipe.InitBuffer(d0OutQueue, 1, tileRow * sizeof(T));
        pipe.InitBuffer(d0OutQueue2, 1, tileRow * sizeof(T));
        pipe.InitBuffer(d0OutQueue3, 1, tileRow * sizeof(T));

        pipe.InitBuffer(d1InQueue, 1, totalColAligned * sizeof(T));
        pipe.InitBuffer(d1OldInQueue, 1, totalColAligned * sizeof(T));
        pipe.InitBuffer(d1OutQueue, 1, totalColAligned * sizeof(T));
        pipe.InitBuffer(d1OutQueue2, 1, totalColAligned * sizeof(T));
        pipe.InitBuffer(d1OutQueue3, 1, totalColAligned * sizeof(T));
        pipe.InitBuffer(cpuSumScratch, CPU_SUM_SCRATCH_SIZE * sizeof(T));

        // 32B对齐
        uint16_t BLOCK_SIZE = 32;
        uint16_t blockLen = totalCol * sizeof(T);
        rowLengthAligned = (((blockLen + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::FillConstantResult(float value)
    {
        for (int t = 0; t < tileNum; t++) {
            uint32_t tileIdx = t * tileLength;
            uint32_t row = (t == tileNum - 1) ? lastTileRow : tileRow;
            LocalTensor<T> resultLocal = costInQueue.AllocTensor<T>();
            for (int r = 0; r < row; r++) {
                Duplicate(resultLocal[r * rowLengthAligned], static_cast<T>(value), totalCol);
            }
            event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
            SaveP<IT>(row, pGlobal[tileIdx], resultLocal);
            costInQueue.FreeTensor(resultLocal);
        }
        event_t eventIdMte3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIdMte3S);
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3S);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::Process()
    {
        if (totalCol == 1) {
            FillConstantResult(1.0f / totalRow);
            return;
        }

        ExpCost();
        InitD();

        while (GetLoopFlag() != 0 && loopCount < MAX_SINKHORN_ITER) {
            loopCount++;
            UpdateD0();
            UpdateD1();
            DUMP_LT_0_2(d0GlobalInWS, totalRow, " d0[glb]");
            DUMP_LT_0_2(d1GlobalInWS, totalCol, " d1[glb]");
        }

        ComputeResult();
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::InitD0GlobalInWS()
    {
        uint32_t tmpBufLength = (totalRow < tileLength) ? totalRow : tileLength;
        if (totalCol > tmpBufLength) {
            tmpBufLength = totalCol;
        }

        LocalTensor<T> tmpLocal = costInQueue.AllocTensor<T>();
        Duplicate(tmpLocal, static_cast<T>(1.0), tmpBufLength);
        // Duplicate(PIPE_V)写UB后，DataCopyPad(PIPE_MTE3)需读取同一UB，补V->MTE3同步
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);

        uint32_t loopCount = totalRow / tmpBufLength;
        uint32_t remainingD0 = totalRow % tmpBufLength;
        for (int i = 0; i < loopCount; i++) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(tmpBufLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(d0GlobalInWS[i * tmpBufLength], tmpLocal, copyParams);
        }
        if (remainingD0 > 0) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainingD0 * sizeof(T)), 0, 0, 0};
            DataCopyPad(d0GlobalInWS[loopCount * tmpBufLength], tmpLocal, copyParams);
        }
        costInQueue.FreeTensor(tmpLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::InitD1GlobalInWS()
    {
        LocalTensor<T> tmpLocal = d1InQueue.AllocTensor<T>();
        Duplicate(tmpLocal, static_cast<T>(1.0), totalCol);
        // Duplicate(PIPE_V)写UB后，DataCopyPad(PIPE_MTE3)需读取同一UB，补V->MTE3同步
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPad(d1GlobalInWS, tmpLocal, copyParams);
        d1InQueue.FreeTensor(tmpLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::InitD1GlobalInWSNew()
    {
        LocalTensor<T> tmpLocal = d1OutQueue.AllocTensor<T>();
        Duplicate(tmpLocal, static_cast<T>(0.0), totalCol);
        // Duplicate(PIPE_V)写UB后，DataCopyPad(PIPE_MTE3)需读取同一UB，补V->MTE3同步
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPad(d1GlobalInWSNew, tmpLocal, copyParams);
        d1OutQueue.FreeTensor(tmpLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::InitD()
    {
        if (blockIdx == 0) {
            // 由block 0执行
            InitD0GlobalInWS(); // costInQueue
            InitD1GlobalInWS(); // d1InQueue
            InitD1GlobalInWSNew(); // d1OutQueue

            SetLoopFlag(1);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(headerInWS);
        }

        SyncAll();

        if (blockIdx != 0) {
            DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(headerInWS);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SetLoopFlag(uint64_t loop)
    {
        headerInWS.SetValue(LOOP_FLAG_INDEX, loop);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(headerInWS);
    }

    template<typename T, typename IT>
    __aicore__ inline uint64_t KernelSinkhorn<T, IT>::GetLoopFlag()
    {
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(headerInWS);
        return headerInWS.GetValue(LOOP_FLAG_INDEX);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeResultCore(
        uint32_t row, LocalTensor<T> d0Local, LocalTensor<T> d1Local)
    {
        LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
        {
            LocalTensor<T> costDstLocal = costOutQueue.AllocTensor<T>();

            // 逐行计算
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                DUMP_LT_2(costSrcLocal[rowIdx], totalCol, "cost: ");
                Mul(costDstLocal[rowIdx], costSrcLocal[rowIdx], d1Local, totalCol);
            }
            costOutQueue.EnQue(costDstLocal);
            costInQueue.FreeTensor(costSrcLocal);
        }
        {
            LocalTensor<T> costDstLocal = costOutQueue.DeQue<T>();
            LocalTensor<T> resultLocal = costInQueue.AllocTensor<T>();

            // 逐行计算
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                DUMP_LT_2(costDstLocal[rowIdx], totalCol, "cost[*d1]: ");
                WaitFlag<HardEvent::V_S>(eventId);
                T d0 = d0Local.GetValue(r);
                Muls(resultLocal[rowIdx], costDstLocal[rowIdx], d0, totalCol);
                SetFlag<HardEvent::V_S>(eventId);
                DUMP_LT_2(resultLocal[rowIdx], totalCol, "cost[*d0]: ");
            }
            WaitFlag<HardEvent::V_S>(eventId);
            costInQueue.EnQue(resultLocal);
            costOutQueue.FreeTensor(costDstLocal);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeResult()
    {
        LocalTensor<T> d1Local = d1InQueue.AllocTensor<T>();

        // copy d1
        DataCacheClean(d1GlobalInWS);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(d1Local, d1GlobalInWS, copyParams, padParams);
        d1InQueue.EnQue(d1Local);

        d1Local = d1InQueue.DeQue<T>();
        DUMP_LT_0_2(d0GlobalInWS, totalRow, " d0[glb]");
        DUMP_LT_0_2(d1Local, totalCol, " d1[glb]");

        event_t eventId_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);

        for (int t = 0; t < tileNum; t++) {
            uint32_t tileIdx = t * this->tileLength;
            uint32_t length = this->tileLength;
            uint32_t row = tileRow;
            if (t == this->tileNum - 1) {
                length = this->lastTileLength;
                row = lastTileRow;
            }

            // 搬入cost
            WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            CopyInFromP<T>(row, pFloatGlobal[tileIdx]);

            LocalTensor<T> d0Local = d0InQueue.AllocTensor<T>();
            DataCopyExtParams d0CopyParams{1, static_cast<uint32_t>(row * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> d0PadParams{false, 0, 0, 0};
            DataCopyPad(d0Local, d0BlockInWS[t * tileRow], d0CopyParams, d0PadParams);
            d0InQueue.EnQue(d0Local);

            // 计算
            d0Local = d0InQueue.DeQue<T>();
            ComputeResultCore(row, d0Local, d1Local);
            d0InQueue.FreeTensor(d0Local);

            // 搬出cost // d1 * cost * d0 融合后，数据在src中
            {
                LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
                SaveP<IT>(row, pGlobal[tileIdx], costSrcLocal);
                costInQueue.FreeTensor(costSrcLocal);
                SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);

        d1InQueue.FreeTensor(d1Local);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::DataCacheClean(GlobalTensor<T> global)
    {
        GM_ADDR p = (GM_ADDR)(global.GetPhyAddr());
        uint32_t dataSize = (GM_ADDR)(global[1].GetPhyAddr()) - p;
        uint32_t stride = 64 / dataSize;
        uint32_t repeatTimes = (global.GetSize() + stride - 1) / stride;

        // 非64B对齐，则多一行
        if (uint64_t(p - 0) % 64) {
            repeatTimes++;
        }
        for (int i = 0; i < repeatTimes; i++) {
            DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global[i * stride]);
        }
    }
} // namespace AscendC

#endif // SINKHORN_KERNEL_INC_
