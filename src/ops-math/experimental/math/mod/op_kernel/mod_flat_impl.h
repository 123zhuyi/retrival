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
 * \file mod_flat_impl.h
 * \brief Mod<T,ST,OT> arch22 flat-buffer (ROUND8/ROUND6) member-function definitions.
 *
 * arch22 flat-buffer 成员定义的纯物理拆分 (同 class / 同 ModNs namespace / 同 MOD_ENH_ARCH22 守卫，行为不变)。
 * 本文件不自带 namespace，从 mod.h 的 `namespace ModNs` 内 #include (类体之后、Init/Process 之前) ->
 * 定义附着到 ModNs::Mod<T,ST,OT>。
 */
#ifndef MOD_FLAT_IMPL_H
#define MOD_FLAT_IMPL_H

#if MOD_ENH_ARCH22
// 为 CONTIGUOUS (isInput2Scalar || isInput2SameShape) 派发分配 FLAT_SLOTS 深 ping-pong buffer + 预分配
// manual event，替代该运行时分支原本的 TQue InitBuffer。尺寸与 TQue 路 1:1 对齐 (FLAT_SLOTS=2 slot，每 slot
// actualMaxDataCount 元素) -> 既有 UB_DIVIDER 预算 (已按双缓冲估算) 直接覆盖，无需改 tiling。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::InitFlatBuffers()
{
    pipe.InitBuffer(flatSelfBuf_, static_cast<uint64_t>(FLAT_SLOTS) * actualMaxDataCount * sizeof(ST));
    if (isInput2Scalar) {
        pipe.InitBuffer(flatOtherScalarBuf_, static_cast<uint64_t>(actualMaxDataCount) * sizeof(OT));
    } else {
        // isInput2SameShape && !isInput2Scalar: genuine per-tile tensor-other read.
        pipe.InitBuffer(flatOtherBuf_, static_cast<uint64_t>(FLAT_SLOTS) * actualMaxDataCount * sizeof(OT));
    }
    pipe.InitBuffer(flatOutBuf_, static_cast<uint64_t>(FLAT_SLOTS) * actualMaxDataCount * sizeof(T));
    for (int32_t s = 0; s < FLAT_SLOTS; ++s) {
        flatEvMte2V_[s] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        flatEvVMte2_[s] = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        flatEvVMte3_[s] = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        flatEvMte3V_[s] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    }
}

template <typename T, typename ST, typename OT>
__aicore__ inline LocalTensor<ST> Mod<T, ST, OT>::FlatSelfSlot(int32_t slot)
{
    return flatSelfBuf_.Get<ST>()[static_cast<uint64_t>(slot) * actualMaxDataCount];
}

template <typename T, typename ST, typename OT>
__aicore__ inline LocalTensor<OT> Mod<T, ST, OT>::FlatOtherSlot(int32_t slot)
{
    return flatOtherBuf_.Get<OT>()[static_cast<uint64_t>(slot) * actualMaxDataCount];
}

template <typename T, typename ST, typename OT>
__aicore__ inline LocalTensor<T> Mod<T, ST, OT>::FlatOutSlot(int32_t slot)
{
    return flatOutBuf_.Get<T>()[static_cast<uint64_t>(slot) * actualMaxDataCount];
}

// 每核一次的标量加载：读 x2[0] (raw OT) 一次，Duplicate 满整 tile 宽度一次。值与原 per-tile CopyIn 的
// isInput2Scalar 分支一致 (DataCopyPad 1 elem -> GetValue -> Duplicate)，仅频率从每 tile 改为每核一次。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::LoadScalarOtherFlat()
{
    LocalTensor<OT> land = flatOtherScalarBuf_.Get<OT>();
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(sizeof(OT));
    p.srcStride = 0;
    p.dstStride = 0;
    DataCopyPad(land, inputx2GM[0], p, {false, 0, 0, 0});
    event_t evMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evMte2S);
    WaitFlag<HardEvent::MTE2_S>(evMte2S);
    OT scalarValue = land.GetValue(0);
    event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evSV);
    WaitFlag<HardEvent::S_V>(evSV);
    Duplicate(land, scalarValue, static_cast<int32_t>(actualMaxDataCount));
}

// 单 slot CopyIn：常见 32B 对齐 tile 用轻量 DataCopy (ROUND6)，真正未对齐 tail 用 DataCopyPad 回退。
// reuse=true 时先等该 slot 上一占用被 compute 消费 (V_MTE2) 再覆写；prologue 预取 tile (reuse=false) 跳过等待。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::CopyInFlat(int32_t slot, uint64_t offset, uint64_t n, bool reuse)
{
    if (reuse) {
        WaitFlag<HardEvent::V_MTE2>(flatEvVMte2_[slot]);
    }
    LocalTensor<ST> selfLocal = FlatSelfSlot(slot);
    constexpr uint64_t alignSelf = 32U / sizeof(ST);
    if ((n % alignSelf) == 0U) {
        DataCopy(selfLocal, inputx1GM[offset], n);
    } else {
        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(n * sizeof(ST));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(selfLocal, inputx1GM[offset], p, {false, 0, 0, 0});
    }
    if (!isInput2Scalar) {
        // isInput2SameShape tensor-other: genuine per-tile read at the SAME offset as self (same shape, no
        // GetInput2Offset needed — 与原 CopyIn 的 isInput2SameShape 分支一致)。
        LocalTensor<OT> otherLocal = FlatOtherSlot(slot);
        constexpr uint64_t alignOther = 32U / sizeof(OT);
        if ((n % alignOther) == 0U) {
            DataCopy(otherLocal, inputx2GM[offset], n);
        } else {
            DataCopyParams p2;
            p2.blockCount = 1;
            p2.blockLen = static_cast<uint32_t>(n * sizeof(OT));
            p2.srcStride = 0;
            p2.dstStride = 0;
            DataCopyPad(otherLocal, inputx2GM[offset], p2, {false, 0, 0, 0});
        }
    }
    SetFlag<HardEvent::MTE2_V>(flatEvMte2V_[slot]);
}

// CopyOut for one flat slot: light DataCopy (ROUND6) for the common 32B-aligned tile, DataCopyPad fallback
// for a genuinely-unaligned tail. Waits V_MTE3 (set by the caller right after ComputeCore) before reading
// the out slot; sets MTE3_V so the caller may reuse this slot FLAT_SLOTS tiles later.
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::CopyOutFlat(int32_t slot, uint64_t offset, uint64_t n)
{
    WaitFlag<HardEvent::V_MTE3>(flatEvVMte3_[slot]);
    LocalTensor<T> outLocal = FlatOutSlot(slot);
    constexpr uint64_t alignOut = 32U / sizeof(T);
    if ((n % alignOut) == 0U) {
        DataCopy(outputGM[offset], outLocal, n);
    } else {
        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(n * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(outputGM[offset], outLocal, p);
    }
    SetFlag<HardEvent::MTE3_V>(flatEvMte3V_[slot]);
}

// CONTIGUOUS 派发的共享 flat-path Process 循环，按编译期 LEAN flag 参数化：flat (ComputeCore，
// int32/int16 same-dtype) 与 lean (ComputeContigLean，fp32/fp16/bf16 same-dtype) 派发共用同一 ping-pong
// 骨架 (避免重复代码)。prologue 预取至 FLAT_SLOTS tile，之后每消费一 tile 预取 t+FLAT_SLOTS 到释放的 slot；
// 两个 drain 循环排空在途 read/out slot。每实例化唯一差异是 per-tile 计算步，由 `if constexpr (LEAN)` 选择，
// 丢弃分支不实例化 -> LEAN=true/LEAN=false 各自等价于原 ProcessContigLean / ProcessContiguousFlat。
template <typename T, typename ST, typename OT>
template <bool LEAN>
__aicore__ inline void Mod<T, ST, OT>::ProcessContigPipeline(uint64_t inOffset, uint64_t outOffset)
{
    const uint64_t loopCount = perCoreDataCount / maxDataCount;
    const uint64_t tailDataCount = perCoreDataCount % maxDataCount;
    const uint64_t totalTiles = loopCount + ((tailDataCount > 0) ? 1U : 0U);
    if (totalTiles == 0) {
        return;
    }
    if (isInput2Scalar) {
        LoadScalarOtherFlat();
    }
    constexpr uint64_t fs = static_cast<uint64_t>(FLAT_SLOTS);
    const uint64_t pre = (totalTiles < fs) ? totalTiles : fs;
    for (uint64_t p = 0; p < pre; ++p) {
        const uint64_t np = (p < loopCount) ? maxDataCount : tailDataCount;
        CopyInFlat(static_cast<int32_t>(p % fs), inOffset + p * maxDataCount, np, /*reuse=*/false);
    }
    for (uint64_t t = 0; t < totalTiles; ++t) {
        const int32_t slot = static_cast<int32_t>(t % fs);
        const uint64_t n = (t < loopCount) ? maxDataCount : tailDataCount;
        // this tile's self/other landed
        WaitFlag<HardEvent::MTE2_V>(flatEvMte2V_[slot]);
        if (t >= fs) {
            WaitFlag<HardEvent::MTE3_V>(flatEvMte3V_[slot]); // out slot's previous occupant drained -> reusable
        }
        LocalTensor<ST> x1Tensor = FlatSelfSlot(slot);
        LocalTensor<OT> x2Tensor = isInput2Scalar ? flatOtherScalarBuf_.Get<OT>() : FlatOtherSlot(slot);
        LocalTensor<T> dstTensor = FlatOutSlot(slot);
        if constexpr (LEAN) {
            ComputeContigLean(static_cast<int32_t>(n), x1Tensor, x2Tensor, dstTensor);
        } else {
            LocalTensor<uint8_t> sharedTmpBuffer = tmpBuff.Get<uint8_t>();
            ComputeCore(static_cast<int32_t>(n), x1Tensor, x2Tensor, dstTensor, sharedTmpBuffer);
        }
        SetFlag<HardEvent::V_MTE2>(flatEvVMte2_[slot]); // read slot reusable by MTE2 (tile t+FLAT_SLOTS)
        SetFlag<HardEvent::V_MTE3>(flatEvVMte3_[slot]); // this out slot ready for MTE3
        CopyOutFlat(slot, outOffset + t * maxDataCount, n);
        const uint64_t tp = t + fs;
        if (tp < totalTiles) {
            const uint64_t npn = (tp < loopCount) ? maxDataCount : tailDataCount;
            CopyInFlat(slot, inOffset + tp * maxDataCount, npn, /*reuse=*/true);
        }
    }
    // drain trailing flags: read slots still in flight (last <=FLAT_SLOTS tiles) + out slots' MTE3_V.
    const uint64_t liveR = (totalTiles < fs) ? totalTiles : fs;
    for (uint64_t s = 0; s < liveR; ++s) {
        WaitFlag<HardEvent::V_MTE2>(flatEvVMte2_[s]);
    }
    const uint64_t liveO = (totalTiles < fs) ? totalTiles : fs;
    for (uint64_t s = 0; s < liveO; ++s) {
        WaitFlag<HardEvent::MTE3_V>(flatEvMte3V_[s]);
    }
}
#endif // MOD_ENH_ARCH22

#endif // MOD_FLAT_IMPL_H
