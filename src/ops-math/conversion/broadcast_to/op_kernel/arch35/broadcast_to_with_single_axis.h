/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_with_single_axis.h
 * \brief 单轴特化模板：单输入 broadcast kernel (静态Tensor编程 + Mutex同步)
 *
 * 静态Tensor编程:
 *   - 使用 LocalMemAllocator 分配UB, 不通过TPipe/TBuf/TQue
 *   - 使用 Mutex (Lock/Unlock) 同步流水线
 *   - A轴: Double Buffer, Ping/Pong交替, 双MutexID
 *   - B轴: 单块常驻UB, NDDMA首次加载后循环搬出
 */

#ifndef BROADCAST_TO_WITH_SINGLE_AXIS_H_
#define BROADCAST_TO_WITH_SINGLE_AXIS_H_

#include "kernel_operator.h"
#include "broadcast_to_with_single_axis_tiling_data.h"

namespace BrcSA {
using namespace AscendC;

constexpr NdDmaConfig copyCfg{false, 0, 0, false};

template <typename T, bool IsBrc>
class BroadcastSingleAxis {
public:
    __aicore__ inline BroadcastSingleAxis(){};
    __aicore__ inline ~BroadcastSingleAxis()
    {
        ReleaseMutexID(mutexId0_);
        ReleaseMutexID(mutexId1_);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ SingleAxisBrcTilingData* tilingDataPtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyScalarAndDup();
    __aicore__ inline void CopyDataIn(int64_t gmOffset, uint32_t curTileSize, LocalTensor<T>& ubBuf);
    __aicore__ inline void CopyDataOut(int64_t gmOffset, uint32_t curTileSize, LocalTensor<T>& ubBuf);

private:
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    // 静态Tensor: LocalMemAllocator分配, 不通过TPipe
    LocalTensor<T> ubPing;
    LocalTensor<T> ubPong;
    LocalTensor<T> scalarBuf; // BRC: 存放从GM读取的标量, 供Duplicate(tensor版)广播

    AscendC::DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};

    // Mutex: 双MutexID用于Double Buffer交替 (参考 matrix_set_diag_v2)
    MutexID mutexId0_{0};
    MutexID mutexId1_{0};

    int64_t blockIdx = 0;
    uint64_t shapeSize = 0;
    uint32_t loopNum = 0;
    uint32_t tileSize = 0;
    uint32_t tileOffset = 0;
};

// ============================================================
// Implementation
// ============================================================

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::Init(GM_ADDR x, GM_ADDR y,
                                                           __tiling_data_ptr__ SingleAxisBrcTilingData* tilingDataPtr)
{
    AscendC::InitSocState();
    tileSize = tilingDataPtr->tileSize;
    shapeSize = tilingDataPtr->shapeSize;

    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    blockIdx = GetBlockIdx() % tilingDataPtr->blockNum;
    uint32_t totalTiles = (tilingDataPtr->shapeSize + tileSize - 1) / tileSize;
    uint32_t mainCoreBlock = tilingDataPtr->blockFactor;
    uint32_t mainCoreNum = totalTiles % tilingDataPtr->blockNum;
    if (mainCoreNum == 0) {
        mainCoreNum = tilingDataPtr->blockNum;
    }
    if (blockIdx < mainCoreNum) {
        loopNum = mainCoreBlock;
        tileOffset = blockIdx * mainCoreBlock;
    } else {
        loopNum = mainCoreBlock - 1;
        tileOffset = mainCoreNum * mainCoreBlock + (blockIdx - mainCoreNum) * (mainCoreBlock - 1);
    }

    // Mutex: 提前分配, 供BRC标量读取的MTE2→V同步复用
    mutexId0_ = AllocMutexID();
    mutexId1_ = AllocMutexID();

    // 静态Tensor: 用LocalMemAllocator分配UB
    LocalMemAllocator<Hardware::UB> ubAlloc;
    if constexpr (IsBrc) {
        ubPing = ubAlloc.Alloc<T>(tileSize);
        scalarBuf = ubAlloc.Alloc<T>(32 / sizeof(T));
    } else {
        ubPing = ubAlloc.Alloc<T>(tileSize);
        ubPong = ubAlloc.Alloc<T>(tileSize);
    }
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CopyScalarAndDup()
{
    // Mutex同步MTE2→V: DataCopyPad(标量)在MTE2流水线, Duplicate在V流水线
    // Duplicate(tensor版): ubPing[i] = scalarBuf[0], 全程V流水线, Mutex可正确同步
    DataCopyExtParams scParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    Mutex::Lock<PIPE_MTE2>(mutexId0_);
    DataCopyPad(scalarBuf, inGM, scParams, copyPadParams);
    Mutex::Unlock<PIPE_MTE2>(mutexId0_);

    Mutex::Lock<PIPE_V>(mutexId0_);
    Duplicate(ubPing, scalarBuf, static_cast<int32_t>(tileSize));
    Mutex::Unlock<PIPE_V>(mutexId0_);
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CopyDataIn(int64_t gmOffset, uint32_t curTileSize,
                                                                 LocalTensor<T>& ubBuf)
{
    DataCopyExtParams params{1, curTileSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(ubBuf, inGM[gmOffset], params, copyPadParams);
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::CopyDataOut(int64_t gmOffset, uint32_t curTileSize,
                                                                  LocalTensor<T>& ubBuf)
{
    DataCopyExtParams params{1, curTileSize * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(outGM[gmOffset], ubBuf, params);
}

template <typename T, bool IsBrc>
__aicore__ inline void BroadcastSingleAxis<T, IsBrc>::Process()
{
    if constexpr (IsBrc) {
        // B轴: Init中GetValue读brcVal → V(Duplicate填充UB) → MTE3循环搬出
        CopyScalarAndDup();

        Mutex::Lock<PIPE_MTE3>(mutexId0_);
        for (uint32_t i = 0; i < loopNum; i++) {
            int64_t gmOffset = (static_cast<int64_t>(tileOffset) + i) * tileSize;
            int64_t remaining = static_cast<int64_t>(shapeSize) - gmOffset;
            uint32_t curTileSize = (remaining < static_cast<int64_t>(tileSize)) ? static_cast<uint32_t>(remaining) :
                                                                                  tileSize;
            CopyDataOut(gmOffset, curTileSize, ubPing);
        }
        Mutex::Unlock<PIPE_MTE3>(mutexId0_);
    } else {
        // A轴: Double Buffer, Ping/Pong交替, 双MutexID
        for (uint32_t i = 0; i < loopNum; i++) {
            int64_t gmOffset = (static_cast<int64_t>(tileOffset) + i) * tileSize;
            int64_t remaining = static_cast<int64_t>(shapeSize) - gmOffset;
            uint32_t curTileSize = (remaining < static_cast<int64_t>(tileSize)) ? static_cast<uint32_t>(remaining) :
                                                                                  tileSize;

            MutexID curMutex = (i & 1) ? mutexId1_ : mutexId0_;
            LocalTensor<T>& curBuf = (i & 1) ? ubPong : ubPing;

            Mutex::Lock<PIPE_MTE2>(curMutex);
            CopyDataIn(gmOffset, curTileSize, curBuf);
            Mutex::Unlock<PIPE_MTE2>(curMutex);

            Mutex::Lock<PIPE_MTE3>(curMutex);
            CopyDataOut(gmOffset, curTileSize, curBuf);
            Mutex::Unlock<PIPE_MTE3>(curMutex);
        }
    }
}

} // namespace BrcSA

#endif // BROADCAST_TO_WITH_SINGLE_AXIS_H_
