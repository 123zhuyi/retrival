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
 * \file mod_algoa_impl.h
 * \brief Mod<T,ST,OT> 的 AlgoA / Dekker 数学辅助成员定义
 *        (CeilAlign / TwoProd / RemainderAlgoA /
 * RemainderAdaptive)。
 * 这些辅助原先内联在 Mod 类体内，现仅在 mod.h 中声明、在此外置定义
 * (与 ComputeInt32 / flat-buffer 方法相同的声明/定义拆分)，纯物理迁移、 行为零变化。本文件不单独包裹 namespace，而是从
 * mod.h 的 `namespace ModNs` 内被 #include，故定义挂接到 `ModNs::Mod<T,ST,OT>`。
 */
#ifndef MOD_ALGOA_IMPL_H
#define MOD_ALGOA_IMPL_H

template <typename T, typename ST, typename OT>
template <typename T1, typename T2>
__aicore__ inline T1 Mod<T, ST, OT>::CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// Dekker TwoProduct (Veltkamp split，全 fp32)：pOut = round(x*y)，eOut = x*y - pOut
// (fp32 乘积的精确残差)。x/y 不变；s0/s1/s2 为与 x/y/pOut/eOut 不相交的暂存块。
// 仅用 Mul/Muls/Sub/Add；split 常量 4097 = 2^12+1 (float 24-bit 尾数)。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::TwoProd(const LocalTensor<float>& pOut, const LocalTensor<float>& eOut,
                                               const LocalTensor<float>& x, const LocalTensor<float>& y,
                                               const LocalTensor<float>& s0, const LocalTensor<float>& s1,
                                               const LocalTensor<float>& s2, int32_t cnt)
{
    constexpr float C = 4097.0f; // 2^12 + 1 Veltkamp split for float
    Mul(pOut, x, y, cnt);        // p = round(x*y)
    Muls(s0, x, C, cnt);         // C*x
    Sub(s1, s0, x, cnt);         // C*x - x
    Sub(s0, s0, s1, cnt);        // hx = C*x - (C*x - x)
    Sub(s1, x, s0, cnt);         // tx = x - hx        (s0=hx, s1=tx)
    Muls(s2, y, C, cnt);         // C*y
    Sub(eOut, s2, y, cnt);       // C*y - y
    Sub(s2, s2, eOut, cnt);      // hy = C*y - (C*y - y)
    Sub(eOut, y, s2, cnt);       // ty = y - hy        (s2=hy, eOut=ty)
    Mul(s0, s0, s2, cnt);        // s0 = hx*hy   (hx no longer needed)
    Sub(s0, s0, pOut, cnt);      // s0 = hx*hy - p
    Mul(s2, s2, s1, cnt);        // s2 = hy*tx   (hy no longer needed)
    Add(s0, s0, s2, cnt);        // s0 += hy*tx
    Mul(eOut, x, eOut, cnt);     // eOut = x*ty  (x=hx+tx exact -> covers hx*ty + tx*ty)
    Add(eOut, s0, eOut, cnt);    // e = (hx*hy - p + hy*tx) + x*ty
}

#if MOD_ENH_ARCH22
// 大商补偿路径：rOut = aIn - trunc(aIn/bIn)*bIn (torch.fmod 大商语义)，aIn/bIn 保留。
// 6 个不相交 fp32 工作块 w0..w5 (均 != rOut/aIn/bIn)；含 Sign 符号修正与 CAST_TRUNC。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::RemainderAlgoA(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                                      const LocalTensor<float>& bIn, const LocalTensor<float>& w0,
                                                      const LocalTensor<float>& w1, const LocalTensor<float>& w2,
                                                      const LocalTensor<float>& w3, const LocalTensor<float>& w4,
                                                      const LocalTensor<float>& w5, int32_t cnt)
{
    // q = trunc(a/b)
    Div(w0, aIn, bIn, cnt);
    Cast(w0, w0, AscendC::RoundMode::CAST_TRUNC, cnt); // w0 = q
    // (hi,lo) = TwoProd(q=w0, b) -> hi=w1, lo=w2 ; split scratch w3,w4,w5 (disjoint from w0/bIn)
    TwoProd(w1, w2, w0, bIn, w3, w4, w5, cnt);
    // r = (a - hi) - lo  (补偿 q*b 的 fp32 乘积舍入残差)
    Sub(rOut, aIn, w1, cnt);
    Sub(rOut, rOut, w2, cnt); // rOut = r = a - q*b (compensated)
    // 廉价约减: k = trunc(r/b); r -= k*b (plain Mul)
    Div(w3, rOut, bIn, cnt);
    Cast(w3, w3, AscendC::RoundMode::CAST_TRUNC, cnt); // w3 = k
    Mul(w3, w3, bIn, cnt);                             // w3 = k*b
    Sub(rOut, rOut, w3, cnt);                          // r -= k*b
    // 符号修正：if sign(r) != sign(a) -> r += sign(a)*|b| (q off-by-1 from 1-ULP Div)。
    Mul(w0, rOut, aIn, cnt);  // w0 = r*a
    Sign(w1, w0, cnt);        // w1 = sign(r*a)
    Muls(w1, w1, -1.0f, cnt); // w1 = -sign(r*a)
    Maxs(w1, w1, 0.0f, cnt);  // w1 = relu(-sign(r*a)) = 1 iff r*a<0 else 0
    Sign(w2, aIn, cnt);       // w2 = sign(a)
    Abs(w3, bIn, cnt);        // w3 = |b|
    Mul(w1, w1, w2, cnt);     // w1 = ind * sign(a)
    Mul(w1, w1, w3, cnt);     // w1 = ind * sign(a) * |b|
    Add(rOut, rOut, w1, cnt); // r += correction
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::RemainderNaive(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                                      const LocalTensor<float>& bIn, const LocalTensor<float>& work,
                                                      int32_t cnt)
{
    Div(work, aIn, bIn, cnt);
    Cast(work, work, AscendC::RoundMode::CAST_TRUNC, cnt);
    Mul(work, work, bIn, cnt);
    Sub(rOut, aIn, work, cnt); // r = a - trunc(a/b)*b (sign follows a)
}

// 自适应路由：per-tile max|a| < naiveThresh_ 走朴素 4-op (小商省算力)，否则走 RemainderAlgoA
// (大商补偿)；tile 内任一 |a|>=thresh 即整 tile 走 AlgoA。maxAbsA = Abs(aIn)->ReduceMax->GetValue。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::RemainderAdaptive(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                                         const LocalTensor<float>& bIn, const LocalTensor<float>& w0,
                                                         const LocalTensor<float>& w1, const LocalTensor<float>& w2,
                                                         const LocalTensor<float>& w3, const LocalTensor<float>& w4,
                                                         const LocalTensor<float>& w5, int32_t cnt)
{
    Abs(w0, aIn, cnt);                        // w0 = |a|
    ReduceMax<float>(w1, w0, w2, cnt, false); // w1[0] = max|a| (w2 = reduce tmp)
    event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evVS);
    WaitFlag<HardEvent::V_S>(evVS); // order ReduceMax(V) -> GetValue(S)
    const float maxAbsA = w1.GetValue(0);
    // NaN: (NaN < thresh) 为 false -> 走 AlgoA (自然传播 NaN, probe verified)。
    if (maxAbsA < naiveThresh_) {
        // NAIVE 4-op (small |a|)
        event_t evSV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evSV0);
        WaitFlag<HardEvent::S_V>(evSV0); // order GetValue(S) -> Div(V) writes w0
        RemainderNaive(rOut, aIn, bIn, w0, cnt);
    } else {
        // big |a|: 32-op Algorithm A
        event_t evSV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evSV1);
        WaitFlag<HardEvent::S_V>(evSV1); // order GetValue(S) -> AlgoA first Div(V) on w0
        RemainderAlgoA(rOut, aIn, bIn, w0, w1, w2, w3, w4, w5, cnt);
    }
}

// ---------------------------------------------------------------------------
// per-core 路由 (0811 深夜设计动机：RemainderAdaptive 的 per-tile 探针 Abs->ReduceMax->GetValue 每 tile
//   付 V_S + S_V 两次跨流水同步 (~1-4us/tile)，大例数千 tile/核时探针税占 50%+ 耗时 -> 曾尝试把路由
//   粒度上卷到 per-core：每核预扫一次 x1 全量 max|a|，整核只路由一次)。
// V5.1 真机 A/B 实证收口 (数据在案)：fp lane 的 per-core 预扫是净亏损——预扫每 chunk ~1.3-1.7us
//   (GM 重读+Cast+Abs+ReduceMax+队列/事件往返)，不低于 per-tile 探针本身 (~0.5-1.5us)，还白付一遍
//   全量 GM 读；实测老套件 52 例回退 (中位 1.36×)、评委套件 26 例回退、052 17.8->19.1ms。
//   故 fp 路保持 per-tile RemainderAdaptive 现状——这是经真机 A/B 证伪后的保留决定，不是没做过。
//   仅保留 int16 锁整核 naive (K2 语义，纯赚：036 -19% / 044 -43% 到 20.41us≈内置 1.23×，
//   044/036 双案例外全绿)；int16 fused 由此免 per-tile 探针且少算 AlgoA。
// ---------------------------------------------------------------------------

// per-core 路由包装：coreRoute_==1 -> 整核 naive 4-op；==2 -> 整核 AlgoA；否则 -> per-tile 现状回落。
//   ==2 分支自 V5.1 起是理论死路 (fp 预扫已实证否决删除，无任何路径置 2)；保留原因 = 对称
//   dispatcher (未来若出现安全的整核 AlgoA 置位来源可直接复用；删掉则本函数退化为单分支包装，
//   反而丢语义)。RemainderNaive/RemainderAlgoA/RemainderAdaptive 的数学主体一行未动。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::RemainderRouted(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                                       const LocalTensor<float>& bIn, const LocalTensor<float>& w0,
                                                       const LocalTensor<float>& w1, const LocalTensor<float>& w2,
                                                       const LocalTensor<float>& w3, const LocalTensor<float>& w4,
                                                       const LocalTensor<float>& w5, int32_t cnt)
{
    if (coreRoute_ == 1U) {
        RemainderNaive(rOut, aIn, bIn, w0, cnt);
    } else if (coreRoute_ == 2U) {
        RemainderAlgoA(rOut, aIn, bIn, w0, w1, w2, w3, w4, w5, cnt);
    } else {
        RemainderAdaptive(rOut, aIn, bIn, w0, w1, w2, w3, w4, w5, cnt); // per-tile 回落 (现状)
    }
}

// 每核一次置位 coreRoute_ (Process() 内 InitConstants 后、派发前调用一次)。V5.1 收敛形态：
//   int32 -> 恒 0 (ComputeInt32 不消费)；int16 -> 1 (K2 整核 naive 锁，纯赚，见上方实证记录)；
//   fp lanes -> 恒 0 (per-core 预扫真机 A/B 证伪净亏，fp 保持 per-tile RemainderAdaptive 现状)。
//   不再做任何 GM 预扫读写。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::PreScanCoreRoute()
{
    if constexpr (std::is_same_v<T, int>) {
        // int32: ComputeInt32 不消费 coreRoute_ -> 恒 0 直接返回。
        return;
    } else if constexpr (!USE_ALGO_A) {
        // K2 语义：int16 全域 naive 精确 (|int16|<=32767<2^24，商<256 距整数边界 >> 半 ulp)，永不 AlgoA
        //   -> 整核 naive (fused 路 int16 由此免 per-tile Abs/ReduceMax 探针且少算 AlgoA)。免预扫。
        coreRoute_ = 1U;
        return;
    } else {
        // fp32/fp16/bf16：保持 0 (per-tile 现状)。per-core fp 预扫已被真机 A/B 证伪净亏
        //   (预扫 chunk 成本 ≳ 探针本身 + 白付一遍全量 GM 读；老套件 52 例/评委 26 例回退在案)，
        //   扫描循环整体删除。
        return;
    }
}
#endif

#endif // MOD_ALGOA_IMPL_H
