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
 * \file mod_bcast_impl.h
 * \brief Mod<T,ST,OT> 融合广播 (fused general-broadcast) 成员定义 (arch22-only)。
 *
 * 融合广播：OUTER 行广播每核只把 x2 的 [INNER] 行读一次 -> UB 常驻复用 (消 mte2 冗余)；INNER 列广播每行
 *   Duplicate (避 GatherMask)。物化广播 divisor 到 fp32，直接跑 RemainderAdaptive -> 精简计算/buffer
 *   (perElem ~44 B/elem)。
 *
 * 隔离：全 `#if MOD_ENH_ARCH22` 守卫；仅当 host 判定资格 (fp32/fp16/bf16/int16 同 dtype + collapse-2D)
 *   置 bcastFusedMode_ != 0 才进入。通用 ProcessBroadcast / mixed / int32 广播 全部走不到融合 (独立运行时
 *   分支、风险隔离)。融合分支自带整套精简 buffer，InitConstants 对融合路跳过 (精简核不用 inf/nan/zero 常量)。
 *   非 arch22 -> 整块编译移除，Process() 回落通用 ProcessBroadcast。
 *
 * 0811 tile 塌陷修复 (评委 0811 复测 Test_036/044 = 24.4×/35.9× 回退的根因)：
 *   原资格要求 INNER 32B 对齐 -> 非对齐 inner (如 95/5) 的 same-dtype 广播 + 全部 int16 广播落官方通用
 *   ProcessBroadcast，tile 被 x2 连续段长卡死 (tile=inner=5/95 元素) -> 每 tile 队列/GM 固定开销不摊薄。
 *   修复 = padding 行布局：UB 内 self/out/otherF32 统一 [rows, bcIpad_] 排布 (bcIpad_ =
 *   ceil(inner*sizeof(dtype)/32)*32/sizeof(dtype)，与 DataCopyPad blockCount 模式自动 padding 落块 lockstep)，
 *   行首恒 32B 对齐 -> 任意 inner 可融合；int16 same-dtype 入列 (fp32 域 RemainderAdaptive 对 int16 输入
 *   精确, 下行 CAST_RINT 与 K2 现用法一致 -> 与通用路 bit 级一致)。pad 车道消毒：x1 两 ping-pong 槽每核
 *   priming 置 0 (防 fp 脏车道 inf/nan 污染 maxAbsA 探针)；x2 恒预填 1.0 (pad 商=0/1=0 良性)；pad 结果被
 *   CopyOut 的 blockLen=inner 丢弃，永不出 UB。
 *
 * 数值等价：精简核直接 RemainderAdaptive 直算 fp32 域，cast 收尾 CAST_RINT。RemainderAdaptive/AlgoA 不改写
 *   aIn/bIn -> OUTER 物化 divisor 常驻块跨 tile 复用安全。inf/nan 由 naive/AlgoA 自然传播。
 *
 * 与 mod_flat_impl.h / mod_compute_impl.h 同惯例：本文件不自带 `namespace ModNs`，从 mod.h 的 namespace 块内
 *   #include -> 定义附着到 ModNs::Mod<T,ST,OT>。
 */
#ifndef MOD_BCAST_IMPL_H
#define MOD_BCAST_IMPL_H

#if MOD_ENH_ARCH22

// 融合广播只对 fp32/fp16/bf16/int16 同 dtype (host 保证)。fp16/bf16/int16 需 cast 到 fp32 域算。
//   NEED_FP32_IO_BUF 对 fp16/bf16/int16 为 true, 复用之判定 cast 分支 (与 same-dtype 计算路一致的 dtype 谓词)。
// 精简 buffer 集 (复用既有 TBuf 成员，融合分支单独 InitBuffer，不与通用路并存)：
//   otherF32 = bcastOtherBuf_ (物化广播 divisor, fp32)；w0..w5 = ResQuot/ResRem/A1..A4；
//   cast: rF32 = A5, selfF32 = x1TensorFP32Buff, cdminF32(小) = x2TensorFP32Buff。
//   perElem(tileAligned 尺度) ~= 44 (fp32/2B 同), + 原始 other 队列/cdmin 小块 (~8*rawAligned) -> ubFormer 更大。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::InitFusedBcastBuffers()
{
    pipe.InitBuffer(inputx1Queue, bufferNum, actualMaxDataCount * sizeof(ST)); // self CopyIn (满 tile 双缓)
    pipe.InitBuffer(outputQueue, bufferNum, actualMaxDataCount * sizeof(T));   // out (满 tile 双缓)
    // 原始 other 读队列: OUTER 广播读 [INNER] 一行; INNER 广播读 [ubFormer] 个 per-row 标量。取二者上界并 64 对齐。
    uint64_t rawElems = (bcastFusedMode_ == 1U) ? bcInner_ : bcUbFormer_;
    uint32_t rawAligned = static_cast<uint32_t>((rawElems + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK);
    if (rawAligned < DATA_BLOCK) {
        rawAligned = DATA_BLOCK;
    }
    pipe.InitBuffer(inputx2Queue, bufferNum, rawAligned * sizeof(OT));
    pipe.InitBuffer(bcastOtherBuf_, actualMaxDataCount * sizeof(float)); // otherF32 (物化广播 divisor, fp32)
    // RemainderAdaptive 的 6 个 disjoint fp32 工作块 (w0..w5)。
    pipe.InitBuffer(ResQuotTensorBuff, maxDataCount * sizeof(float)); // w0
    pipe.InitBuffer(ResRemTensorBuff, maxDataCount * sizeof(float));  // w1
    pipe.InitBuffer(A1Buff, maxDataCount * sizeof(float));            // w2
    pipe.InitBuffer(A2Buff, maxDataCount * sizeof(float));            // w3
    pipe.InitBuffer(A3Buff, maxDataCount * sizeof(float));            // w4
    pipe.InitBuffer(A4Buff, maxDataCount * sizeof(float));            // w5
    if constexpr (NEED_FP32_IO_BUF) { // fp16/bf16/int16: cast self + rF32 结果块 + cdmin 小 fp32 scratch
        pipe.InitBuffer(A5Buff, maxDataCount * sizeof(float));           // rF32 (cast 结果)
        pipe.InitBuffer(x1TensorFP32Buff, maxDataCount * sizeof(float)); // selfF32 (cast target)
        pipe.InitBuffer(x2TensorFP32Buff, rawAligned * sizeof(float));   // cdminF32 (原始 other -> fp32, 小)
    }
}

// 遍历本核负责的 OUTER 行, 逐 tile CopyIn(self) -> Compute -> CopyOut。OUTER 行广播的物化 divisor 只建一次
// (BuildOuterOtherFused, 常驻复用 -> 每 tile 零 other GM 流量, 即消 mte2 冗余的关键杠杆); INNER 列广播在
// ComputeFusedBcast 内逐 tile 由 per-row 标量 Duplicate 重建。
// 0811: 进主循环前 priming —— x1 两个 ping-pong 物理槽整槽 Duplicate(0) (两次 outstanding Alloc 锁定两个
//   槽位)。此后 2D 自动 padding 拷入只写真实车道, pad 车道恒 0 -> maxAbsA 探针 (ReduceMax|aIn|) 不被脏车道
//   污染 (fp 脏车道可带 inf/nan 位型; nan 会把需 AlgoA 的 tile 误判 naive -> 精度风险, 必须消毒)。
//   PIPE_ALL 排干 priming 的 V 写, 保证后续 MTE2 CopyIn 安全复用槽位。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ProcessFusedBcast()
{
    if (coreRows_ == 0) {
        return; // 越界/空转核 (OUTER 行数 < needCoreNum 时后段核无行)
    }
    {
        LocalTensor<ST> p0 = inputx1Queue.AllocTensor<ST>();
        LocalTensor<ST> p1 = inputx1Queue.AllocTensor<ST>();
        Duplicate(p0, static_cast<ST>(0), static_cast<int32_t>(actualMaxDataCount));
        Duplicate(p1, static_cast<ST>(0), static_cast<int32_t>(actualMaxDataCount));
        inputx1Queue.EnQue(p0);
        inputx1Queue.EnQue(p1);
        LocalTensor<ST> d0 = inputx1Queue.DeQue<ST>();
        LocalTensor<ST> d1 = inputx1Queue.DeQue<ST>();
        inputx1Queue.FreeTensor(d0);
        inputx1Queue.FreeTensor(d1);
        PipeBarrier<PIPE_ALL>();
    }
    if (bcastFusedMode_ == 1U) {
        BuildOuterOtherFused(); // OUTER 行广播: 每核只读 x2 的 [INNER] 行一次 -> fp32 复制成 [ubFormer*bcIpad_] 常驻块
    }
    uint64_t rowOffset = coreRowBase_;
    uint64_t rowsLeft = coreRows_;
    while (rowsLeft > 0) {
        const uint64_t rows = (rowsLeft < bcUbFormer_) ? rowsLeft : bcUbFormer_;
        CopyInFusedBcast(rowOffset, rows);
        ComputeFusedBcast(rows);
        CopyOutFusedBcast(rowOffset, rows);
        rowOffset += rows;
        rowsLeft -= rows;
    }
}

// OUTER 行广播 (other=[1,INNER]): 每核把 x2 的单个 [INNER] 广播行读一次, 复制成 [ubFormer*bcIpad_] 的 fp32
// 常驻块 (otherF32)。此后每 tile 直接复用, 零 per-tile other 流量 (= 反 mte2 冗余杠杆)。
//   fp32: 原始行即 fp32, VIEW 后逐子行 V 域复制 (Muls x1.0)。
//   cast(fp16/bf16/int16)：先 Cast(CAST_NONE) 原始 2B 行 -> cdminF32 (精确 widen，同 flat cast 路的 other widen)，
//                    再逐子行复制 fp32。r*bcIpad_*4B 因 bcIpad_ 32B 对齐 -> 每子行首址 32B 对齐, V 写合法。
//   0811: 复制前整块预填 1.0 -> pad 车道除数恒 1.0 (x1 pad=0 -> 商=0 良性)；逐行复制用 Muls(_,1.0f)
//   (V 域, 任意 inner 靠 mask 合法, 保 -0/NaN/inf 位型; 原 UB->UB DataCopy 要求 count 32B 对齐, 非对齐不可用)。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::BuildOuterOtherFused()
{
    LocalTensor<OT> otherLocal = inputx2Queue.AllocTensor<OT>();
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(bcInner_ * sizeof(OT));
    p.srcStride = 0;
    p.dstStride = 0;
    DataCopyPad(otherLocal, inputx2GM[0], p, {false, 0, 0, 0}); // 单个 [INNER] 广播行
    inputx2Queue.EnQue(otherLocal);
    LocalTensor<OT> otherIn = inputx2Queue.DeQue<OT>();
    LocalTensor<float> otherF32 = bcastOtherBuf_.Get<float>();
    const int32_t innerCnt = static_cast<int32_t>(bcInner_);
    LocalTensor<float> srcRow;
    if constexpr (NEED_FP32_IO_BUF) {
        srcRow = x2TensorFP32Buff.Get<float>();                // cdminF32 (小)
        Cast(srcRow, otherIn, RoundMode::CAST_NONE, innerCnt); // 2B [INNER] -> fp32 (精确)
        PipeBarrier<PIPE_ALL>();                               // 排序 Cast(V) -> 下方逐行 Muls(V) (once-per-core)
    } else {
        srcRow = otherIn.template ReinterpretCast<float>(); // fp32-native: 直接 VIEW 原始行
    }
    // pad 车道预填 1.0 (x1 pad=0 -> 商=0 良性; 防 0/0=nan 污染) -> 再逐行覆写真实车道。
    Duplicate(otherF32, 1.0f, static_cast<int32_t>(bcUbFormer_ * bcIpad_));
    for (uint64_t r = 0; r < bcUbFormer_; ++r) {
        Muls(otherF32[r * bcIpad_], srcRow, 1.0f, innerCnt); // V 域行复制 (每核只付一次)
    }
    inputx2Queue.FreeTensor(otherIn);
    PipeBarrier<PIPE_ALL>(); // 排空这一次性复制, 再让首 tile 的 compute 读 otherF32
}

// CopyIn self [rows, INNER] -> UB [rows, bcIpad_] padding 行布局 (DataCopyPad blockCount 模式自动按
//   align32(INNER*sizeof(ST)) 落块, 与 host bcIpad_ lockstep; pad 车道保持 primed 0)。INNER 列广播
//   (other=[OUTER,1]) 额外读本 tile 的 [rows] 个 per-row 标量 (稍后在 ComputeFusedBcast 内逐行 Duplicate
//   成 fp32); OUTER 行广播的 divisor 已常驻, 此处不读。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::CopyInFusedBcast(uint64_t rowOffset, uint64_t rows)
{
    LocalTensor<ST> selfLocal = inputx1Queue.AllocTensor<ST>();
    DataCopyParams sp;
    sp.blockCount = static_cast<uint16_t>(rows); // rows <= bcUbFormer_ << 4095 上限
    sp.blockLen = static_cast<uint32_t>(bcInner_ * sizeof(ST));
    sp.srcStride = 0; // GM 侧行连续
    sp.dstStride = 0; // UB 侧自动 padding 到 32B 块 (行步长 == bcIpad_)
    DataCopyPad(selfLocal, inputx1GM[rowOffset * bcInner_], sp, {false, 0, 0, 0});
    inputx1Queue.EnQue(selfLocal);

    if (bcastFusedMode_ == 2U) {
        LocalTensor<OT> otherLocal = inputx2Queue.AllocTensor<OT>();
        DataCopyParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(rows * sizeof(OT));
        op.srcStride = 0;
        op.dstStride = 0;
        DataCopyPad(otherLocal, inputx2GM[rowOffset], op, {false, 0, 0, 0}); // [rows] 个 per-row 标量
        inputx2Queue.EnQue(otherLocal);
    }
}

// 精简 fp32 域核：物化 divisor 到 otherF32 后跑 RemainderAdaptive 直算。
//   INNER: 逐 OUTER 行把该行标量 Duplicate 满 INNER 列 (V op, PIPE_V 排序)；OUTER: otherF32 已常驻。
//   self: fp32-native VIEW 输入；cast(fp16/bf16/int16) Cast(CAST_NONE) -> selfF32。
//   out : fp32-native 直写 outLocal(VIEW)；cast Cast(CAST_RINT) rF32 -> outLocal。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ComputeFusedBcast(uint64_t rows)
{
    const int32_t cnt = static_cast<int32_t>(rows * bcIpad_); // padding 行布局全车道 (pad 已消毒: x1=0/x2=1)
    LocalTensor<ST> selfLocal = inputx1Queue.DeQue<ST>();
    LocalTensor<float> otherF32 = bcastOtherBuf_.Get<float>();

    if (bcastFusedMode_ == 2U) {
        // INNER 列广播: 逐 OUTER 行 Duplicate 标量满 INNER。fp32 直接读 lane; cast 先 Cast 整片 [rows] -> fp32。
        LocalTensor<OT> otherLocal = inputx2Queue.DeQue<OT>();
        const int32_t innerCnt = static_cast<int32_t>(bcInner_);
        LocalTensor<float> otherScalarF32;
        if constexpr (NEED_FP32_IO_BUF) {
            otherScalarF32 = x2TensorFP32Buff.Get<float>(); // cdminF32 (小)
            Cast(otherScalarF32, otherLocal, RoundMode::CAST_NONE, static_cast<int32_t>(rows));
            event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evVS);
            WaitFlag<HardEvent::V_S>(evVS); // 排序 Cast(V) -> GetValue(S)
        } else {
            event_t evMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(evMte2S);
            WaitFlag<HardEvent::MTE2_S>(evMte2S); // 排序 CopyIn(MTE2) -> GetValue(S)
            otherScalarF32 = otherLocal.template ReinterpretCast<float>();
        }
        // 0811: 先整 tile 预填 1.0 (pad 车道除数非零 -> 商=0 良性) 再逐行覆写真实车道 (行步长 bcIpad_,
        //   首址 32B 对齐 -> 任意 inner 的 Duplicate 合法)。
        Duplicate(otherF32, 1.0f, cnt);
        for (uint64_t r = 0; r < rows; ++r) {
            const float sc = otherScalarF32.GetValue(r);
            Duplicate(otherF32[r * bcIpad_], sc, innerCnt); // 32B-safe 标量广播 (无 GatherMask)
        }
        inputx2Queue.FreeTensor(otherLocal);
        PipeBarrier<PIPE_V>(); // 排序 Duplicate(V) -> Div(V)
    }

    // self 到 fp32
    LocalTensor<float> selfF32;
    if constexpr (NEED_FP32_IO_BUF) {
        selfF32 = x1TensorFP32Buff.Get<float>();
        Cast(selfF32, selfLocal, RoundMode::CAST_NONE, cnt); // fp16/bf16 -> fp32 (精确 widen)
    } else {
        selfF32 = selfLocal.template ReinterpretCast<float>();
    }

    // RemainderAdaptive 6 个 disjoint 工作块。
    LocalTensor<float> w0 = ResQuotTensorBuff.Get<float>();
    LocalTensor<float> w1 = ResRemTensorBuff.Get<float>();
    LocalTensor<float> w2 = A1Buff.Get<float>();
    LocalTensor<float> w3 = A2Buff.Get<float>();
    LocalTensor<float> w4 = A3Buff.Get<float>();
    LocalTensor<float> w5 = A4Buff.Get<float>();

    LocalTensor<T> outLocal = outputQueue.AllocTensor<T>();
    if constexpr (NEED_FP32_IO_BUF) {
        // fp16/bf16/int16: r 落 A5，再 Cast(CAST_RINT) 回 T。
        LocalTensor<float> rF32 = A5Buff.Get<float>();
        RemainderRouted(rF32, selfF32, otherF32, w0, w1, w2, w3, w4, w5, cnt);
        Cast(outLocal, rF32, RoundMode::CAST_RINT, cnt);
    } else {
        // fp32-native: r 直写 output slot (VIEW)。
        LocalTensor<float> outF32 = outLocal.template ReinterpretCast<float>();
        RemainderRouted(outF32, selfF32, otherF32, w0, w1, w2, w3, w4, w5, cnt);
    }

    inputx1Queue.FreeTensor(selfLocal);
    outputQueue.EnQue(outLocal);
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::CopyOutFusedBcast(uint64_t rowOffset, uint64_t rows)
{
    LocalTensor<T> dstLocal = outputQueue.DeQue<T>();
    DataCopyParams cp;
    cp.blockCount = static_cast<uint16_t>(rows);
    cp.blockLen = static_cast<uint32_t>(bcInner_ * sizeof(T)); // 只写真实车道, pad 结果丢弃
    cp.srcStride = 0; // UB 侧自动按 align32(blockLen) 跳过 pad (= bcIpad_ 行步长)
    cp.dstStride = 0; // GM 侧行连续
    DataCopyPad(outputGM[rowOffset * bcInner_], dstLocal, cp);
    outputQueue.FreeTensor(dstLocal);
}

#endif // MOD_ENH_ARCH22

#endif // MOD_BCAST_IMPL_H
