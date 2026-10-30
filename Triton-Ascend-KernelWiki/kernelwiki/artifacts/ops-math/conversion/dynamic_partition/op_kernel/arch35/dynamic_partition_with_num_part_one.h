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
 * \file dynamic_partition_with_num_part_one.h
 * \brief impl of DynamicPartition when num_partitions=1 (pure DMA copy path)
 *
 * 流水线设计：
 *   使用 TQueBind<VECIN, VECOUT, 2> 将 CopyIn 和 CopyOut 绑定到同一块 buffer，
 *   depth=2 开启双缓冲，MTE2（GM→UB）与 MTE3（UB→GM）自动流水并行：
 *     tile i:   [MTE2_i ][MTE3_i ]
 *     tile i+1:          [MTE2_i+1][MTE3_i+1]
 *   相比原方案（单 TQue<VECIN>），消除了 MTE2/MTE3 串行等待，带宽利用率翻倍。
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_WITH_NUM_PART_ONE_H_
#define OP_KERNEL_DYNAMIC_PARTITION_WITH_NUM_PART_ONE_H_

#include "dynamic_partition_tiling_data_struct.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace DynPart {
using namespace AscendC;

template <typename T>
class DynPartWithNumPartOne {
public:
    __aicore__ inline DynPartWithNumPartOne(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                const DynPartNumPartOneTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyProcess();
    __aicore__ inline void WriteOutputShape();

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    const DynPartNumPartOneTilingData* tdPtr_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t blockIdx_ = 0;

    GlobalTensor<T> xInGM_;
    ListTensorDesc outGMList_;
    GlobalTensor<T> xOutGM_;
    GlobalTensor<uint64_t> yShapeGM_;

    // VECIN 与 VECOUT 绑定同一块 buffer：
    //   - AllocTensor / DataCopyPad(GM→UB) / EnQue  → MTE2 搬入路径
    //   - DeQue / DataCopyPad(UB→GM) / FreeTensor   → MTE3 搬出路径
    // depth=2 开启 double buffer，两条通路自动流水
    TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_TWO> queBind_;
    // 专用于写出 yshape 的小 UB buffer（仅最后一核使用）
    // 大小：SHAPE_GAP(9) × sizeof(uint64_t) = 72 字节，对齐到 32 字节 → 96 字节
    TBuf<QuePosition::VECCALC> shapeBuf_;
};

template <typename T>
__aicore__ inline void DynPartWithNumPartOne<T>::Init(GM_ADDR x, GM_ADDR /*partitions*/, GM_ADDR y, GM_ADDR yshape,
                                                      const DynPartNumPartOneTilingData* tilingDataPtr, TPipe* pipeIn)
{
    tdPtr_ = tilingDataPtr;
    pipe_ = pipeIn;
    blockIdx_ = AscendC::GetBlockIdx();

    xInGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGMList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));
    xOutGM_.SetGlobalBuffer(outGMList_.GetDataPtr<T>(0)); // 唯一输出分区 y[0]
    yShapeGM_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(yshape));

    // 每个 slot 大小 = ubFactor 个元素，depth=2 分配两个 slot 实现 double buffer
    uint32_t bufSize = static_cast<uint32_t>(tdPtr_->ubFactor * sizeof(T));
    pipe_->InitBuffer(queBind_, NUM_TWO, bufSize);
    // 为 yshape 写出预留 UB：SHAPE_GAP(9) × sizeof(uint64_t) = 72B，对齐到 32B → 96B
    constexpr uint32_t SHAPE_BUF_SIZE = 96U;
    pipe_->InitBuffer(shapeBuf_, SHAPE_BUF_SIZE);
}

template <typename T>
__aicore__ inline void DynPartWithNumPartOne<T>::CopyProcess()
{
    int64_t elemSize = (blockIdx_ == tdPtr_->usedCoreCnt - 1) ? tdPtr_->tailSize : tdPtr_->mainSize;
    int64_t offset = blockIdx_ * tdPtr_->mainSize;

    int64_t loopNum = elemSize / tdPtr_->ubFactor;
    int64_t ubTailFactor = elemSize % tdPtr_->ubFactor;

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams mainCopyParams{1, static_cast<uint32_t>(tdPtr_->ubFactor * sizeof(T)), 0, 0, 0};

    // 主循环（整 ubFactor 块）
    // tile i 的 MTE2 与 tile i-1 的 MTE3 自动流水（由 TQueBind depth=2 保证）
    for (int64_t i = 0; i < loopNum; ++i) {
        int64_t gmOffset = offset + i * tdPtr_->ubFactor;

        auto ubX = queBind_.AllocTensor<T>();
        DataCopyPad(ubX, xInGM_[gmOffset], mainCopyParams, padParams); // MTE2: GM → UB
        queBind_.EnQue(ubX);

        ubX = queBind_.DeQue<T>();
        DataCopyPad(xOutGM_[gmOffset], ubX, mainCopyParams); // MTE3: UB → GM
        queBind_.FreeTensor(ubX);
    }

    // 尾块（剩余 ubTailFactor 个元素）
    if (ubTailFactor > 0) {
        int64_t gmOffset = offset + loopNum * tdPtr_->ubFactor;
        DataCopyExtParams tailCopyParams{1, static_cast<uint32_t>(ubTailFactor * sizeof(T)), 0, 0, 0};

        auto ubX = queBind_.AllocTensor<T>();
        DataCopyPad(ubX, xInGM_[gmOffset], tailCopyParams, padParams); // MTE2: GM → UB
        queBind_.EnQue(ubX);

        ubX = queBind_.DeQue<T>();
        DataCopyPad(xOutGM_[gmOffset], ubX, tailCopyParams); // MTE3: UB → GM
        queBind_.FreeTensor(ubX);
    }
}

template <typename T>
__aicore__ inline void DynPartWithNumPartOne<T>::WriteOutputShape()
{
    // 只由最后一核写 yshape，与 base.h::RefreshOutputShapes 行为一致
    if (blockIdx_ != tdPtr_->usedCoreCnt - 1) {
        return;
    }
    // yshape 格式（SHAPE_GAP=9 个 uint64 per partition）：
    //   [0]: dimNum | B64_FLAG（标识 dtype 为 uint64）
    //   [1]: dim0（partitions 元素总数）
    //   [2..dimNum]: outDimsExtFirst（partition 以外的维度大小）
    //   其余: 填 1
    constexpr uint32_t SHAPE_GAP = 9U;
    constexpr uint64_t B64_FLAG = (0x1UL << 31);

    auto ubShape = shapeBuf_.Get<uint64_t>();
    Duplicate(ubShape, 1UL, static_cast<int32_t>(SHAPE_GAP));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    uint64_t dimNum = static_cast<uint64_t>(tdPtr_->dimNumExtFirst + 1);
    ubShape.SetValue(0, dimNum | B64_FLAG);
    ubShape.SetValue(1, static_cast<uint64_t>(tdPtr_->dim0));
    for (int64_t i = 0; i < tdPtr_->dimNumExtFirst; ++i) {
        ubShape.SetValue(static_cast<uint32_t>(i + 2), static_cast<uint64_t>(tdPtr_->outDimsExtFirst[i]));
    }

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(SHAPE_GAP * sizeof(uint64_t)), 0, 0, 0};
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(yShapeGM_[0], ubShape, copyParams);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

template <typename T>
__aicore__ inline void DynPartWithNumPartOne<T>::Process()
{
    CopyProcess();
    WriteOutputShape();
}

} // namespace DynPart

#endif // OP_KERNEL_DYNAMIC_PARTITION_WITH_NUM_PART_ONE_H_
