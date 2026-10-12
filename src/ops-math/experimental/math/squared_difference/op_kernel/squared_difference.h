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
 * \file squared_difference.h
 * \brief SquaredDifference 算子 kernel 实现
 *        y = (x1 - x2)^2；OneDim + BRC 双分支；fp32/fp16/bf16/int32/int64
 */

#ifndef SQUAREDDIFFERENCE_H
#define SQUAREDDIFFERENCE_H

#include "kernel_operator.h"
#include "adv_api/pad/broadcast.h"
#include "squared_difference_tiling_data.h"
#include "squared_difference_tiling_key.h"

using namespace AscendC;

constexpr int64_t SD_COPY_MAX_REPEAT = 255;

// T  = 存储 dtype；CT = 计算 dtype（half/bf16 -> float，其余 == T）
// NEED_CAST = 是否升精度（half/bf16 为 true）
template <typename T, typename CT, bool NEED_CAST>
class KernelSquaredDifference {
public:
    __aicore__ inline KernelSquaredDifference(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const SquaredDifferenceTilingData* t)
    {
        tiling_ = t;
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        maxTile_ = t->maxTileElem;

        if (t->mode == SD_MODE_ONEDIM) {
            pipe_->InitBuffer(inQ1_, kQDepth, maxTile_ * sizeof(T));
            pipe_->InitBuffer(inQ2_, kQDepth, maxTile_ * sizeof(T));
            pipe_->InitBuffer(outQ_, kQDepth, maxTile_ * sizeof(T));
            if constexpr (NEED_CAST) {
                pipe_->InitBuffer(tf1_, maxTile_ * sizeof(CT));
                pipe_->InitBuffer(tf2_, maxTile_ * sizeof(CT));
            }
        } else {
            pipe_->InitBuffer(stgQ1_, kQDepth, maxTile_ * sizeof(T));
            int64_t x2Elems = (t->brcKind != 0 || t->bcastOnM != 0) ? t->srcTileElems : maxTile_;
            pipe_->InitBuffer(stgQ2_, kQDepth, x2Elems * sizeof(T));
            pipe_->InitBuffer(outQB_, kQDepth, maxTile_ * sizeof(T));
            if constexpr (!std::is_same<T, int64_t>::value) {
                pipe_->InitBuffer(work1_, maxTile_ * sizeof(T));
                if (t->brcKind != 0) {
                    // fast path：只用 work1_（brcExp）+ brcBcastTmp_（广播展开 tmp）
                    int64_t elemPer32B = 32 / sizeof(T);
                    int64_t brcTmpElems;
                    if (t->brcKind == 2) {
                        // tmp 放大到覆盖整个 tile，使系统 Broadcast 一次循环完成（否则按 16 行分 7 次，21 op + 21
                        // barrier）
                        int64_t numBlocksAlign = ((t->innerDim + elemPer32B - 1) / elemPer32B) * elemPer32B;
                        int64_t minTmp = elemPer32B * elemPer32B + elemPer32B * numBlocksAlign;
                        int64_t repeats = (t->ubFormer + elemPer32B - 1) / elemPer32B;
                        brcTmpElems = repeats * minTmp;
                    } else {
                        int64_t alignInnerSrc = ((t->innerSrc + elemPer32B - 1) / elemPer32B) * elemPer32B;
                        brcTmpElems = t->ubFormer * t->broadcastLen * alignInnerSrc;
                    }
                    pipe_->InitBuffer(brcBcastTmp_, brcTmpElems * sizeof(T));
                } else {
                    pipe_->InitBuffer(work2_, maxTile_ * sizeof(T));
                    pipe_->InitBuffer(brcTmp_, maxTile_ * sizeof(T));
                }
            }
            if constexpr (NEED_CAST) {
                pipe_->InitBuffer(x1Buf_, maxTile_ * sizeof(CT));
                pipe_->InitBuffer(x2Buf_, maxTile_ * sizeof(CT));
            }
        }
    }

    __aicore__ inline void Process()
    {
        // int64 与其它 dtype 走同一搬运/广播框架；DAV_2201 向量指令不支持 int64，
        // 故其计算/填充走 LocalTensor 标量路径（见 ComputeSquaredDiff / FillScalar），
        // 并在 ProcessOneDim / ProcessBrc 中为标量读写补 MTE2->S / S->MTE3 同步。
        // 全程仅用无版本宏保护的稳定原语（DataCopyPad + LocalTensor 标量 + SetFlag/WaitFlag），
        // 兼容 CANN 8.5.0（GM 标量 GetValue/SetValue 是 super-kernel 专属，8.5.0 无）。
        if (tiling_->mode == SD_MODE_ONEDIM) {
            ProcessOneDim();
        } else {
            if constexpr (std::is_same<T, int64_t>::value) {
                ProcessBrc();
            } else {
                if (tiling_->brcKind != 0) {
                    ProcessBrcFast();
                } else {
                    ProcessBrc();
                }
            }
        }
    }

    // 搬入(MTE2)后、标量 GetValue(S) 前等待；fp16 仅在确实标量读时需要。
    __aicore__ inline void SyncMte2ToScalar(bool needHalfScalarRead = false)
    {
        if constexpr (std::is_same<T, int64_t>::value) {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(e);
            WaitFlag<HardEvent::MTE2_S>(e);
        } else if constexpr (std::is_same<T, half>::value) {
            if (needHalfScalarRead) {
                TEventID e = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
                SetFlag<HardEvent::MTE2_S>(e);
                WaitFlag<HardEvent::MTE2_S>(e);
            }
        }
    }

    // 标量路径：标量 SetValue(S) 后、搬出(MTE3) 前，等 S 完成
    __aicore__ inline void SyncScalarToMte3()
    {
        if constexpr (std::is_same<T, int64_t>::value) {
            TEventID e = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
            SetFlag<HardEvent::S_MTE3>(e);
            WaitFlag<HardEvent::S_MTE3>(e);
        }
    }

private:
    static constexpr int32_t kQDepth = 2;

    TPipe* pipe_;
    const SquaredDifferenceTilingData* tiling_;
    GlobalTensor<T> x1Gm_, x2Gm_, yGm_;
    int64_t maxTile_;

    TQue<TPosition::VECIN, kQDepth> inQ1_, inQ2_;
    TQue<TPosition::VECOUT, kQDepth> outQ_;
    TBuf<TPosition::VECCALC> tf1_, tf2_;

    TQue<TPosition::VECIN, kQDepth> stgQ1_, stgQ2_;
    TQue<TPosition::VECOUT, kQDepth> outQB_;
    TBuf<TPosition::VECCALC> work1_, work2_;
    TBuf<TPosition::VECCALC> brcTmp_;
    TBuf<TPosition::VECCALC> brcBcastTmp_;
    TBuf<TPosition::VECCALC> x1Buf_, x2Buf_;

    // 广播展开：src[curOuter*innerSrc] -> dst[curOuter*inner]，inner = innerSrc*broadcastLen，紧凑无 padding。
    // kind=2（尾维，innerSrc==1）：系统 LastDim 广播（Brcb+Copy+GatherMask）。
    __aicore__ inline void ExpandBrcLastDim(const LocalTensor<T>& dst, const LocalTensor<T>& src, int64_t curOuter,
                                            int64_t broadcastLen)
    {
        uint32_t dstShape[2] = {(uint32_t)curOuter, (uint32_t)broadcastLen};
        uint32_t srcShape[2] = {(uint32_t)curOuter, 1};
        LocalTensor<uint8_t> tmp = brcBcastTmp_.template Get<uint8_t>();
        AscendC::Broadcast<T, 2, 1>(dst, src, dstShape, srcShape, tmp);
    }

    // kind=1（中间维）展开：srcPadded[curOuter,alignInnerSrc] -> dst[curOuter,inner]。
    // 对齐 TBE 多核实现：mask 设一次，vcopy 逐外层重复到独立区域，最后一次性 vreducev2(GatherMask)。
    __aicore__ inline void ExpandBrcMiddle(const LocalTensor<T>& dst, const LocalTensor<T>& srcPadded,
                                           const LocalTensor<T>& repeated, int64_t curOuter, int64_t innerSrc,
                                           int64_t broadcastLen, int64_t alignInnerSrc)
    {
        constexpr int64_t elemPer32B = 32 / sizeof(T);
        uint16_t alignBlocks = (uint16_t)(alignInnerSrc / elemPer32B);
        int64_t repStride = broadcastLen * alignInnerSrc;

        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>((uint32_t)alignInnerSrc);
        for (int64_t o = 0; o < curOuter; o++) {
            Copy<T, false>(repeated[o * repStride], srcPadded[o * alignInnerSrc], MASK_PLACEHOLDER,
                           (uint8_t)broadcastLen, {1, 1, alignBlocks, 0});
        }
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();

        GatherMaskParams gp{1, (uint16_t)(curOuter * broadcastLen), alignBlocks, 0};
        uint64_t rsvdCnt = 0;
        GatherMask(dst, repeated, 7, true, (uint32_t)innerSrc, gp, rsvdCnt);
        SetMaskCount();
        PipeBarrier<PIPE_V>();
    }

    // BRC 快速路径：为 tile ui 发起 x1/x2 的搬入（异步 MTE2），供流水线预取。
    __aicore__ inline void IssueFastCopyIn(int64_t ui, int64_t inner, int64_t innerSrc, bool isLastDim,
                                           const GlobalTensor<T>& fullGm, const GlobalTensor<T>& brcGm)
    {
        int64_t outerStart = ui * tiling_->ubFormer;
        int64_t curOuter = tiling_->ubFormer;
        if (outerStart + curOuter > tiling_->outerTotal)
            curOuter = tiling_->outerTotal - outerStart;
        if (curOuter <= 0)
            return;
        int64_t fullElems = curOuter * inner;
        int64_t offFull = outerStart * inner;
        int64_t offBrc = outerStart * innerSrc;

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        LocalTensor<T> fullL = stgQ1_.template AllocTensor<T>();
        DataCopyExtParams pf{1, (uint32_t)(fullElems * sizeof(T)), 0, 0, 0};
        DataCopyPad(fullL, fullGm[offFull], pf, pad);
        stgQ1_.EnQue(fullL);

        LocalTensor<T> srcL = stgQ2_.template AllocTensor<T>();
        if (isLastDim) {
            int64_t srcElems = curOuter * innerSrc;
            DataCopyExtParams ps{1, (uint32_t)(srcElems * sizeof(T)), 0, 0, 0};
            DataCopyPad(srcL, brcGm[offBrc], ps, pad);
        } else {
            DataCopyExtParams ps{(uint16_t)curOuter, (uint32_t)(innerSrc * sizeof(T)), 0, 0, 0};
            DataCopyPad(srcL, brcGm[offBrc], ps, pad);
        }
        stgQ2_.EnQue(srcL);
    }

    // BRC 快速路径（单轴广播）：inner 合并广播轴，split 轴前移；x1 连续搬入、x2 只搬源、
    // 紧凑展开、输出连续搬出（对齐 TBE 多核实现）。
    // 流水线：下一 tile 的搬入(MTE2) 与当前 tile 的计算(V) 重叠，隐藏搬入延迟。
    __aicore__ inline void ProcessBrcFast()
    {
        int64_t inner = tiling_->innerDim;
        int64_t innerSrc = tiling_->innerSrc;
        int64_t broadcastLen = tiling_->broadcastLen;
        int64_t outerTotal = tiling_->outerTotal;
        int64_t outerTile = tiling_->ubFormer;
        bool x1Brc = (tiling_->brcWhich == 1);
        bool isLastDim = (tiling_->brcKind == 2);
        constexpr int64_t elemPer32B = 32 / sizeof(T);

        const GlobalTensor<T>& fullGm = x1Brc ? x2Gm_ : x1Gm_;
        const GlobalTensor<T>& brcGm = x1Brc ? x1Gm_ : x2Gm_;

        int64_t unitStart, unitCount;
        CoreRange(unitStart, unitCount);
        if (unitCount <= 0)
            return;

        // 预取第一个 tile 的搬入
        IssueFastCopyIn(unitStart, inner, innerSrc, isLastDim, fullGm, brcGm);

        for (int64_t i = 0; i < unitCount; i++) {
            int64_t ui = unitStart + i;
            int64_t outerStart = ui * outerTile;
            int64_t curOuter = outerTile;
            if (outerStart + curOuter > outerTotal)
                curOuter = outerTotal - outerStart;
            if (curOuter <= 0)
                break;

            int64_t fullElems = curOuter * inner;
            int64_t offFull = outerStart * inner;

            // 预取下一 tile 的搬入（与当前 tile 计算重叠）
            if (i + 1 < unitCount) {
                IssueFastCopyIn(ui + 1, inner, innerSrc, isLastDim, fullGm, brcGm);
            }

            LocalTensor<T> brcExp;
            if (isLastDim) {
                // kind=2：x2 源搬入 stgQ2_，再系统 LastDim 广播到 work1_
                LocalTensor<T> src = stgQ2_.template DeQue<T>();
                brcExp = work1_.template Get<T>();
                ExpandBrcLastDim(brcExp, src, curOuter, broadcastLen);
                stgQ2_.FreeTensor(src);
            } else {
                // kind=1：x2 用 block 搬入（copy_gm_to_ubuf_align 自动按 32B 对齐补 pad），再重复+GatherMask 紧凑
                int64_t alignInnerSrc = ((innerSrc + elemPer32B - 1) / elemPer32B) * elemPer32B;
                LocalTensor<T> srcPadded = stgQ2_.template DeQue<T>();
                brcExp = work1_.template Get<T>();
                LocalTensor<T> repeated = brcBcastTmp_.template Get<T>();
                ExpandBrcMiddle(brcExp, srcPadded, repeated, curOuter, innerSrc, broadcastLen, alignInnerSrc);
                stgQ2_.FreeTensor(srcPadded);
            }

            LocalTensor<T> full = stgQ1_.template DeQue<T>();
            LocalTensor<T> yL = outQB_.template AllocTensor<T>();
            LocalTensor<CT> f0, f1;
            if constexpr (NEED_CAST) {
                f0 = x1Buf_.Get<CT>();
                f1 = x2Buf_.Get<CT>();
            }

            ComputeSquaredDiff(yL, full, brcExp, f0, f1, (int32_t)fullElems);

            outQB_.template EnQue<T>(yL);
            stgQ1_.FreeTensor(full);

            LocalTensor<T> out = outQB_.template DeQue<T>();
            DataCopyExtParams po{1, (uint32_t)(fullElems * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offFull], out, po);
            outQB_.FreeTensor(out);
        }
    }

    __aicore__ inline void CoreRange(int64_t& start, int64_t& count)
    {
        const int64_t blockIdx = GetBlockIdx();
        const int64_t base = tiling_->blockBase;
        const int64_t remainder = tiling_->blockRemainder;
        count = base + (blockIdx < remainder ? 1 : 0);
        start = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);
    }

    template <bool SCALAR_X1, bool SCALAR_X2>
    __aicore__ inline void ComputeOneDimInt64(const LocalTensor<T>& dst, const LocalTensor<T>& a,
                                              const LocalTensor<T>& b, int64_t count)
    {
        if constexpr (SCALAR_X1 && SCALAR_X2) {
            int64_t d = (a.GetValue(0) - b.GetValue(0));
            int64_t dd = d * d;
            for (int64_t k = 0; k < count; k++)
                dst.SetValue(k, dd);
        } else if constexpr (SCALAR_X1) {
            int64_t va = a.GetValue(0);
            int64_t k = 0;
            for (; k + 3 < count; k += 4) {
                int64_t d0 = va - b.GetValue(k);
                int64_t d1 = va - b.GetValue(k + 1);
                int64_t d2 = va - b.GetValue(k + 2);
                int64_t d3 = va - b.GetValue(k + 3);
                dst.SetValue(k, d0 * d0);
                dst.SetValue(k + 1, d1 * d1);
                dst.SetValue(k + 2, d2 * d2);
                dst.SetValue(k + 3, d3 * d3);
            }
            for (; k < count; k++) {
                int64_t d = va - b.GetValue(k);
                dst.SetValue(k, d * d);
            }
        } else if constexpr (SCALAR_X2) {
            int64_t vb = b.GetValue(0);
            int64_t k = 0;
            for (; k + 3 < count; k += 4) {
                int64_t d0 = a.GetValue(k) - vb;
                int64_t d1 = a.GetValue(k + 1) - vb;
                int64_t d2 = a.GetValue(k + 2) - vb;
                int64_t d3 = a.GetValue(k + 3) - vb;
                dst.SetValue(k, d0 * d0);
                dst.SetValue(k + 1, d1 * d1);
                dst.SetValue(k + 2, d2 * d2);
                dst.SetValue(k + 3, d3 * d3);
            }
            for (; k < count; k++) {
                int64_t d = a.GetValue(k) - vb;
                dst.SetValue(k, d * d);
            }
        } else {
            int64_t k = 0;
            for (; k + 3 < count; k += 4) {
                int64_t d0 = a.GetValue(k) - b.GetValue(k);
                int64_t d1 = a.GetValue(k + 1) - b.GetValue(k + 1);
                int64_t d2 = a.GetValue(k + 2) - b.GetValue(k + 2);
                int64_t d3 = a.GetValue(k + 3) - b.GetValue(k + 3);
                dst.SetValue(k, d0 * d0);
                dst.SetValue(k + 1, d1 * d1);
                dst.SetValue(k + 2, d2 * d2);
                dst.SetValue(k + 3, d3 * d3);
            }
            for (; k < count; k++) {
                int64_t d = a.GetValue(k) - b.GetValue(k);
                dst.SetValue(k, d * d);
            }
        }
    }

    __aicore__ inline void ComputeOneDimInt64Dispatch(const LocalTensor<T>& dst, const LocalTensor<T>& a,
                                                      const LocalTensor<T>& b, int64_t count, bool scX1, bool scX2)
    {
        if (scX1) {
            if (scX2)
                ComputeOneDimInt64<true, true>(dst, a, b, count);
            else
                ComputeOneDimInt64<true, false>(dst, a, b, count);
        } else {
            if (scX2)
                ComputeOneDimInt64<false, true>(dst, a, b, count);
            else
                ComputeOneDimInt64<false, false>(dst, a, b, count);
        }
    }

    // float → bf16 舍入（round-to-nearest-even），返回 bf16-representable 的 float。
    // 整数位操作实现，全程无 bf16 算术（后端不支持 bf16 标量）。
    __aicore__ inline float RoundToBf16(float v)
    {
        uint32_t bits;
        __builtin_memcpy(&bits, &v, 4);
        uint32_t exp = (bits >> 23) & 0xFFu;
        if (exp == 0xFFu) { // NaN/Inf：直接截断，不进位
            bits &= 0xFFFF0000u;
        } else {
            uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7FFFu + lsb; // RNE 进位偏置
            bits &= 0xFFFF0000u;   // 截断低 16 位（bf16 尾数 7 位）
        }
        float r;
        __builtin_memcpy(&r, &bits, 4);
        return r;
    }

    __aicore__ inline void ComputeSquaredDiff(const LocalTensor<T>& dst, const LocalTensor<T>& a,
                                              const LocalTensor<T>& b, const LocalTensor<CT>& f0,
                                              const LocalTensor<CT>& f1, int32_t count)
    {
        if constexpr (std::is_same<T, int64_t>::value) {
            // DAV_2201 向量指令不支持 int64，走标量逐元素计算 (x1-x2)^2。
            // 调用方保证减法和平方均在 int64 可表示范围内。
            for (int32_t k = 0; k < count; k++) {
                int64_t d = a.GetValue(k) - b.GetValue(k);
                dst.SetValue(k, d * d);
            }
        } else if constexpr (std::is_same<T, half>::value) {
            // fp16 向量 2-round：原生 half Sub(round1) → PipeBarrier 阻止与 Mul 融合 → half Mul(round2)。
            Sub(dst, a, b, count);
            PipeBarrier<PIPE_V>();
            Mul(dst, dst, dst, count);
        } else if constexpr (NEED_CAST) {
            // bf16 向量 2-round：mask 设一次，7 个 op 复用，减少标量 mask 开销。
            // Cast(bf16→fp32) → Sub → Cast(fp32→bf16, round1) → PipeBarrier
            // → Cast(bf16→fp32) → Mul → Cast(fp32→bf16, round2)。
            SetMaskCount();
            SetVectorMask<CT, MaskMode::COUNTER>((uint32_t)count);
            Cast<CT, T, false>(f0, a, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2});
            Cast<CT, T, false>(f1, b, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2});
            Sub<CT, false>(f0, f0, f1, MASK_PLACEHOLDER, 1, BinaryRepeatParams{});
            Cast<T, CT, false>(dst, f0, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE});
            PipeBarrier<PIPE_V>();
            Cast<CT, T, false>(f0, dst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2});
            Mul<CT, false>(f0, f0, f0, MASK_PLACEHOLDER, 1, BinaryRepeatParams{});
            Cast<T, CT, false>(dst, f0, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1,
                               {1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE});
            SetMaskNorm();
            ResetMask();
        } else {
            Sub(dst, a, b, count);
            Mul(dst, dst, dst, count);
        }
    }

    __aicore__ inline void ProcessOneDim()
    {
        int64_t ubFormer = tiling_->ubFormer;
        int64_t unitStart, unitCount;
        CoreRange(unitStart, unitCount);
        int64_t offset = unitStart * ubFormer;
        bool scX1 = tiling_->x1Scalar != 0;
        bool scX2 = tiling_->x2Scalar != 0;

        for (int64_t i = 0; i < unitCount; i++) {
            int64_t unit = unitStart + i;
            int64_t curLen = (unit == tiling_->ubOuter - 1) ? tiling_->ubTail : ubFormer;

            LocalTensor<T> x1L = inQ1_.template AllocTensor<T>();
            LocalTensor<T> x2L = inQ2_.template AllocTensor<T>();
            CopyInOne(x1L, x1Gm_, scX1, offset, curLen);
            CopyInOne(x2L, x2Gm_, scX2, offset, curLen);
            inQ1_.EnQue(x1L);
            inQ2_.EnQue(x2L);

            LocalTensor<T> a = inQ1_.template DeQue<T>();
            LocalTensor<T> b = inQ2_.template DeQue<T>();
            SyncMte2ToScalar(scX1 || scX2);
            LocalTensor<T> yL = outQ_.template AllocTensor<T>();
            LocalTensor<CT> f0, f1;
            if constexpr (NEED_CAST) {
                f0 = tf1_.Get<CT>();
                f1 = tf2_.Get<CT>();
            }
            if constexpr (std::is_same<T, int64_t>::value) {
                ComputeOneDimInt64Dispatch(yL, a, b, curLen, scX1, scX2);
            } else {
                if (scX1 && scX2 && !NEED_CAST) {
                    CT d = static_cast<CT>(a.GetValue(0)) - static_cast<CT>(b.GetValue(0));
                    FillScalar(yL, static_cast<T>(d * d), curLen);
                } else {
                    if (scX1) {
                        T s = a.GetValue(0);
                        FillScalar(a, s, curLen);
                    }
                    if (scX2) {
                        T s = b.GetValue(0);
                        FillScalar(b, s, curLen);
                    }
                    ComputeSquaredDiff(yL, a, b, f0, f1, curLen);
                }
            }
            outQ_.template EnQue<T>(yL);
            inQ1_.FreeTensor(a);
            inQ2_.FreeTensor(b);

            LocalTensor<T> out = outQ_.template DeQue<T>();
            SyncScalarToMte3(); // int64：等标量写完成再搬出
            DataCopyExtParams p{1, (uint32_t)(curLen * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offset], out, p);
            outQ_.FreeTensor(out);
            offset += ubFormer;
        }
    }

    __aicore__ inline void CopyInOne(const LocalTensor<T>& dst, const GlobalTensor<T>& gm, bool scalar, int64_t offset,
                                     int64_t curLen)
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if (scalar) {
            DataCopyExtParams p{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(dst, gm[0], p, pad);
        } else {
            DataCopyExtParams p{1, (uint32_t)(curLen * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst, gm[offset], p, pad);
        }
    }

    __aicore__ inline int BrcKind(int64_t mStride, int64_t nStride)
    {
        if (mStride != 0 && nStride != 0)
            return 0;
        if (mStride == 0 && nStride != 0)
            return 1;
        if (mStride != 0 && nStride == 0)
            return 2;
        return 3;
    }

    // int64 广播（广播轴=M，内轴 N 切分）：整广播轴一个 tile，读 x1[M,curN]、x2[curN]，
    // 标量跨 M 广播（kind2=1），消除外层广播的冗余搬移与冗余 tile。
    __aicore__ inline void ProcessBrcInt64Bcast()
    {
        int64_t M = tiling_->ubFormer;
        int64_t inner = tiling_->innerDim;
        int64_t nFormer = tiling_->nFormer;
        int64_t nOuter = tiling_->nOuter;
        int64_t bAxis = tiling_->ubSplitAxis;
        bool x1Brc = (tiling_->brcWhich == 1);

        const GlobalTensor<T>& fullGm = x1Brc ? x2Gm_ : x1Gm_;
        const GlobalTensor<T>& brcGm = x1Brc ? x1Gm_ : x2Gm_;

        int64_t unitStart, unitCount;
        CoreRange(unitStart, unitCount);
        if (unitCount <= 0)
            return;

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        int64_t alignN = nFormer;

        for (int64_t loop = 0; loop < unitCount; loop++) {
            int64_t u = unitStart + loop;
            int64_t nTile = u % nOuter;
            int64_t outerCombo = u / nOuter;

            int64_t idx[SD_MAX_DIM] = {0};
            int64_t rem = outerCombo;
            for (int64_t i = bAxis - 1; i >= 0; i--) {
                idx[i] = rem % tiling_->outDims[i];
                rem /= tiling_->outDims[i];
            }

            int64_t nBase = nTile * nFormer;
            int64_t curN = (nTile == nOuter - 1) ? (inner - nBase) : nFormer;

            int64_t outerOff = 0;
            for (int64_t i = 0; i < bAxis; i++)
                outerOff += idx[i] * tiling_->outStrides[i];

            LocalTensor<T> sFull = stgQ1_.template AllocTensor<T>();
            LocalTensor<T> sBrc = stgQ2_.template AllocTensor<T>();
            for (int64_t m = 0; m < M; m++) {
                DataCopyExtParams p{1, (uint32_t)(curN * sizeof(T)), 0, 0, 0};
                DataCopyPad(sFull[m * alignN], fullGm[outerOff + m * inner + nBase], p, pad);
            }
            {
                DataCopyExtParams p{1, (uint32_t)(curN * sizeof(T)), 0, 0, 0};
                DataCopyPad(sBrc, brcGm[outerOff + nBase], p, pad);
            }
            stgQ1_.EnQue(sFull);
            stgQ2_.EnQue(sBrc);

            LocalTensor<T> a = stgQ1_.template DeQue<T>();
            LocalTensor<T> b = stgQ2_.template DeQue<T>();
            SyncMte2ToScalar();
            LocalTensor<T> yL = outQB_.template AllocTensor<T>();
            ComputeBrcInt64Dispatch(yL, a, b, M, curN, alignN, 0, 1);
            outQB_.template EnQue<T>(yL);
            stgQ1_.FreeTensor(a);
            stgQ2_.FreeTensor(b);

            LocalTensor<T> out = outQB_.template DeQue<T>();
            SyncScalarToMte3();
            for (int64_t m = 0; m < M; m++) {
                DataCopyExtParams p{1, (uint32_t)(curN * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm_[outerOff + m * inner + nBase], out[m * alignN], p);
            }
            outQB_.FreeTensor(out);
        }
    }

    // 独立 N 切分路径：仅 nOuter>1（单行超 UB 预算）时使用，M-tile 固定=1。
    // 每 unit = (外层组合, M 行, N-tile)，逐块搬入 curN 列、计算、搬出。完全不依赖
    // ProcessBrc 的 M 切分逻辑，避免影响 nOuter==1 的已验证路径。
    __aicore__ inline void ProcessBrcNSplit()
    {
        int64_t unitStart, unitCount;
        CoreRange(unitStart, unitCount);
        int64_t ubSplitAxis = tiling_->ubSplitAxis;
        int64_t Mdim = tiling_->outDims[ubSplitAxis];
        int64_t N = tiling_->innerDim;
        int64_t nFormer = tiling_->nFormer;
        int64_t nOuter = tiling_->nOuter;
        int64_t shapeLen = tiling_->shapeLen;
        int64_t nAxis = shapeLen - 1;
        int kind1 = BrcKind(tiling_->x1Strides[ubSplitAxis], tiling_->x1Strides[nAxis]);
        int kind2 = BrcKind(tiling_->x2Strides[ubSplitAxis], tiling_->x2Strides[nAxis]);
        int64_t x1nStride = tiling_->x1Strides[nAxis];
        int64_t x2nStride = tiling_->x2Strides[nAxis];
        int64_t x1mStride = tiling_->x1Strides[ubSplitAxis];
        int64_t x2mStride = tiling_->x2Strides[ubSplitAxis];

        // 外层组合总数 = prod(outDims[0..ubSplitAxis-1])
        int64_t outerProd = 1;
        for (int64_t i = 0; i < ubSplitAxis; i++)
            outerProd *= tiling_->outDims[i];

        for (int64_t loop = 0; loop < unitCount; loop++) {
            // 线性 unit 分解为 (outerCombo, mRow, nTile)，nTile 最快变化
            int64_t u = unitStart + loop;
            int64_t nTileIdx = u % nOuter;
            int64_t u2 = u / nOuter;
            int64_t mRow = u2 % Mdim;
            int64_t outerCombo = u2 / Mdim;

            int64_t idx[SD_MAX_DIM] = {0};
            int64_t rem = outerCombo;
            for (int64_t i = ubSplitAxis - 1; i >= 0; i--) {
                idx[i] = rem % tiling_->outDims[i];
                rem /= tiling_->outDims[i];
            }

            int64_t nBase = nTileIdx * nFormer;
            int64_t curN = (nTileIdx == nOuter - 1) ? (N - nBase) : nFormer;

            int64_t off1 = 0, off2 = 0, offY = 0;
            for (int64_t i = 0; i < ubSplitAxis; i++) {
                off1 += idx[i] * tiling_->x1Strides[i];
                off2 += idx[i] * tiling_->x2Strides[i];
                offY += idx[i] * tiling_->outStrides[i];
            }
            off1 += mRow * x1mStride + nBase * x1nStride;
            off2 += mRow * x2mStride + nBase * x2nStride;
            offY += mRow * tiling_->outStrides[ubSplitAxis] + nBase;

            LocalTensor<T> s1 = stgQ1_.template AllocTensor<T>();
            LocalTensor<T> s2 = stgQ2_.template AllocTensor<T>();
            // M=1 单行：kind0/1 搬 curN 真实列；kind2/3 搬 1 元素后 V 侧/标量展开
            StageRow(s1, x1Gm_, off1, curN, kind1);
            StageRow(s2, x2Gm_, off2, curN, kind2);
            stgQ1_.EnQue(s1);
            stgQ2_.EnQue(s2);

            LocalTensor<T> a = stgQ1_.template DeQue<T>();
            LocalTensor<T> b = stgQ2_.template DeQue<T>();
            SyncMte2ToScalar(kind1 >= 2 || kind2 >= 2);
            LocalTensor<T> yL = outQB_.template AllocTensor<T>();
            LocalTensor<CT> f0, f1;
            if constexpr (NEED_CAST) {
                f0 = x1Buf_.Get<CT>();
                f1 = x2Buf_.Get<CT>();
            }
            if constexpr (std::is_same<T, int64_t>::value) {
                ComputeBrcInt64Dispatch(yL, a, b, 1, curN, curN, kind1, kind2);
            } else {
                LocalTensor<T> opA = a, opB = b;
                if (kind1 >= 2) {
                    LocalTensor<T> w = work1_.Get<T>();
                    T v = a.GetValue(0);
                    FillScalarOrDup(w, v, curN);
                    opA = w;
                }
                if (kind2 >= 2) {
                    LocalTensor<T> w = work2_.Get<T>();
                    T v = b.GetValue(0);
                    FillScalarOrDup(w, v, curN);
                    opB = w;
                }
                ComputeSquaredDiff(yL, opA, opB, f0, f1, (int32_t)curN);
            }
            outQB_.template EnQue<T>(yL);
            stgQ1_.FreeTensor(a);
            stgQ2_.FreeTensor(b);

            LocalTensor<T> out = outQB_.template DeQue<T>();
            SyncScalarToMte3();
            DataCopyExtParams p{1, (uint32_t)(curN * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offY], out, p);
            outQB_.FreeTensor(out);
        }
    }

    // N 切分路径的单行搬入：非广播搬 curN 列，广播搬 1 元素
    __aicore__ inline void StageRow(const LocalTensor<T>& dst, const GlobalTensor<T>& gm, int64_t off, int64_t curN,
                                    int kind)
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if (kind == 0 || kind == 1) {
            DataCopyExtParams p{1, (uint32_t)(curN * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst, gm[off], p, pad);
        } else {
            DataCopyExtParams p{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(dst, gm[off], p, pad);
        }
    }

    // 填充 curN 个 v：int64 标量 SetValue，其余 Duplicate
    __aicore__ inline void FillScalarOrDup(const LocalTensor<T>& dst, T v, int64_t curN)
    {
        if constexpr (std::is_same<T, int64_t>::value) {
            for (int64_t i = 0; i < curN; i++)
                dst.SetValue(i, v);
        } else {
            Duplicate<T>(dst, v, (int32_t)curN);
        }
    }

    __aicore__ inline void ProcessBrc()
    {
        // int64 广播（广播轴=M，内轴 N 切分）：读广播源一次、标量跨 M 广播
        if constexpr (std::is_same<T, int64_t>::value) {
            if (tiling_->bcastOnM != 0) {
                ProcessBrcInt64Bcast();
                return;
            }
        }
        // N 单行超 UB 预算时走独立 N 切分路径（不影响 nOuter==1 的已验证逻辑）
        if (tiling_->nOuter > 1) {
            ProcessBrcNSplit();
            return;
        }

        int64_t unitStart, unitCount;
        CoreRange(unitStart, unitCount);
        int64_t ubSplitAxis = tiling_->ubSplitAxis;
        int64_t ubOuter = tiling_->ubOuter;
        int64_t ubFormer = tiling_->ubFormer;
        int64_t N = tiling_->innerDim;
        int64_t alignN = tiling_->alignInner;
        int64_t shapeLen = tiling_->shapeLen;
        int kind1 = BrcKind(tiling_->x1Strides[ubSplitAxis], tiling_->x1Strides[shapeLen - 1]);
        int kind2 = BrcKind(tiling_->x2Strides[ubSplitAxis], tiling_->x2Strides[shapeLen - 1]);

        int64_t idx[SD_MAX_DIM] = {0};
        int64_t flat = unitStart;
        int64_t divs[SD_MAX_DIM];
        int64_t acc = ubOuter;
        divs[ubSplitAxis] = 1;
        for (int64_t i = ubSplitAxis - 1; i >= 0; i--) {
            divs[i] = acc;
            acc *= tiling_->outDims[i];
        }
        for (int64_t i = 0; i < ubSplitAxis; i++) {
            idx[i] = flat / divs[i];
            flat %= divs[i];
        }
        idx[ubSplitAxis] = flat;

        // 预取首 tile 的搬入
        int64_t off1 = 0, off2 = 0, offY = 0, curM = 0;
        ComputeBrcOffsets(idx, ubSplitAxis, ubOuter, ubFormer, off1, off2, offY, curM);
        {
            LocalTensor<T> s1 = stgQ1_.template AllocTensor<T>();
            LocalTensor<T> s2 = stgQ2_.template AllocTensor<T>();
            Stage(s1, x1Gm_, off1, curM, N, alignN, kind1, tiling_->x1Strides[ubSplitAxis]);
            Stage(s2, x2Gm_, off2, curM, N, alignN, kind2, tiling_->x2Strides[ubSplitAxis]);
            stgQ1_.EnQue(s1);
            stgQ2_.EnQue(s2);
        }

        for (int64_t loop = 0; loop < unitCount; loop++) {
            // 预取下一 tile 的搬入（MTE2 与当前 tile 标量计算重叠）
            int64_t nOff1 = 0, nOff2 = 0, nOffY = 0, nCurM = 0;
            if (loop + 1 < unitCount) {
                AdvanceIdx(idx, ubSplitAxis, ubOuter);
                ComputeBrcOffsets(idx, ubSplitAxis, ubOuter, ubFormer, nOff1, nOff2, nOffY, nCurM);
                LocalTensor<T> ns1 = stgQ1_.template AllocTensor<T>();
                LocalTensor<T> ns2 = stgQ2_.template AllocTensor<T>();
                Stage(ns1, x1Gm_, nOff1, nCurM, N, alignN, kind1, tiling_->x1Strides[ubSplitAxis]);
                Stage(ns2, x2Gm_, nOff2, nCurM, N, alignN, kind2, tiling_->x2Strides[ubSplitAxis]);
                stgQ1_.EnQue(ns1);
                stgQ2_.EnQue(ns2);
            }

            LocalTensor<T> a = stgQ1_.template DeQue<T>();
            LocalTensor<T> b = stgQ2_.template DeQue<T>();
            bool needHalfScalarRead = (kind1 == 3 || kind2 == 3 || (kind1 == 2 && (curM % 8) != 0) ||
                                       (kind2 == 2 && (curM % 8) != 0));
            SyncMte2ToScalar(needHalfScalarRead);
            LocalTensor<T> yL = outQB_.template AllocTensor<T>();
            LocalTensor<CT> f0, f1;
            if constexpr (NEED_CAST) {
                f0 = x1Buf_.Get<CT>();
                f1 = x2Buf_.Get<CT>();
            }
            if constexpr (std::is_same<T, int64_t>::value) {
                ComputeBrcInt64Dispatch(yL, a, b, curM, N, alignN, kind1, kind2);
            } else {
                int64_t padEle = curM * alignN;
                if (kind1 == 3 && kind2 == 3 && !NEED_CAST) {
                    CT d = static_cast<CT>(a.GetValue(0)) - static_cast<CT>(b.GetValue(0));
                    FillScalar(yL, static_cast<T>(d * d), padEle);
                } else {
                    LocalTensor<T> opA = a, opB = b;
                    LocalTensor<T> tmp = brcTmp_.Get<T>();
                    if (kind1 != 0) {
                        LocalTensor<T> w = work1_.Get<T>();
                        Expand(w, a, tmp, curM, alignN, kind1);
                        opA = w;
                    }
                    if (kind2 != 0) {
                        LocalTensor<T> w = work2_.Get<T>();
                        Expand(w, b, tmp, curM, alignN, kind2);
                        opB = w;
                    }
                    ComputeSquaredDiff(yL, opA, opB, f0, f1, (int32_t)padEle);
                }
            }
            outQB_.template EnQue<T>(yL);
            stgQ1_.FreeTensor(a);
            stgQ2_.FreeTensor(b);

            LocalTensor<T> out = outQB_.template DeQue<T>();
            SyncScalarToMte3(); // int64：等标量写完成再搬出
            if (alignN == N) {
                DataCopyExtParams p{1, (uint32_t)(curM * N * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm_[offY], out, p);
            } else {
                DataCopyExtParams p{(uint16_t)curM, (uint32_t)(N * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm_[offY], out, p);
            }
            outQB_.FreeTensor(out);

            off1 = nOff1;
            off2 = nOff2;
            offY = nOffY;
            curM = nCurM;
        }
    }

    __aicore__ inline void ComputeBrcOffsets(const int64_t idx[SD_MAX_DIM], int64_t ubSplitAxis, int64_t ubOuter,
                                             int64_t ubFormer, int64_t& off1, int64_t& off2, int64_t& offY,
                                             int64_t& curM)
    {
        int64_t tileIdx = idx[ubSplitAxis];
        curM = (tileIdx == ubOuter - 1) ? tiling_->ubTail : ubFormer;
        int64_t mBase = tileIdx * ubFormer;
        off1 = 0;
        off2 = 0;
        offY = 0;
        for (int64_t i = 0; i < ubSplitAxis; i++) {
            off1 += idx[i] * tiling_->x1Strides[i];
            off2 += idx[i] * tiling_->x2Strides[i];
            offY += idx[i] * tiling_->outStrides[i];
        }
        off1 += mBase * tiling_->x1Strides[ubSplitAxis];
        off2 += mBase * tiling_->x2Strides[ubSplitAxis];
        offY += mBase * tiling_->outStrides[ubSplitAxis];
    }

    __aicore__ inline void Stage(const LocalTensor<T>& dst, const GlobalTensor<T>& gm, int64_t off, int64_t M,
                                 int64_t N, int64_t alignN, int kind, int64_t mStride)
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if (kind == 0) {
            if (alignN == N) {
                DataCopyExtParams p{1, (uint32_t)(M * N * sizeof(T)), 0, 0, 0};
                DataCopyPad(dst, gm[off], p, pad);
            } else {
                DataCopyExtParams p{(uint16_t)M, (uint32_t)(N * sizeof(T)), 0, 0, 0};
                DataCopyPad(dst, gm[off], p, pad);
            }
        } else if (kind == 1) {
            DataCopyExtParams p{1, (uint32_t)(N * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst, gm[off], p, pad);
        } else if (kind == 2) {
            DataCopyExtParams p{1, (uint32_t)(M * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst, gm[off], p, pad);
        } else {
            DataCopyExtParams p{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(dst, gm[off], p, pad);
        }
    }

    // 向量 Duplicate 填充；int64 不支持向量指令，走标量 SetValue 逐元素填充
    __aicore__ inline void FillScalar(const LocalTensor<T>& dst, T v, int64_t count)
    {
        if constexpr (std::is_same<T, int64_t>::value) {
            for (int64_t i = 0; i < count; i++)
                dst.SetValue(i, v);
        } else {
            Duplicate<T>(dst, v, (int32_t)count);
        }
    }

    __aicore__ inline void Expand(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& tmp,
                                  int64_t M, int64_t alignN, int kind)
    {
        if (kind == 1) {
            SetMaskCount();
            SetVectorMask<T, MaskMode::COUNTER>((uint32_t)alignN);
            uint16_t rowBlocks = (uint16_t)(alignN * sizeof(T) / SD_UB_BLOCK_SIZE);
            int64_t rowsDone = 0;
            while (rowsDone < M) {
                int64_t rowsLeft = M - rowsDone;
                uint8_t repeat = (uint8_t)(rowsLeft > SD_COPY_MAX_REPEAT ? SD_COPY_MAX_REPEAT : rowsLeft);
                Copy<T, false>(dst[rowsDone * alignN], src, MASK_PLACEHOLDER, repeat, {1, 1, rowBlocks, 0});
                rowsDone += repeat;
            }
            SetMaskNorm();
            ResetMask();
        } else if (kind == 2) {
            constexpr int64_t oneBlockElem = SD_UB_BLOCK_SIZE / sizeof(T);
            int64_t fullRows = (M / 8) * 8;
            int64_t brcbRepeats = fullRows / 8;
            int64_t repeatsDone = 0;
            while (repeatsDone < brcbRepeats) {
                int64_t repeatsLeft = brcbRepeats - repeatsDone;
                uint8_t repeat = (uint8_t)(repeatsLeft > SD_COPY_MAX_REPEAT ? SD_COPY_MAX_REPEAT : repeatsLeft);
                Brcb(tmp[repeatsDone * 8 * oneBlockElem], src[repeatsDone * 8], repeat, {1, DEFAULT_REPEAT_STRIDE});
                repeatsDone += repeat;
            }
            if (fullRows > 0) {
                PipeBarrier<PIPE_V>();
                SetMaskCount();
                SetVectorMask<T, MaskMode::COUNTER>((uint32_t)alignN);
                uint16_t rowBlocks = (uint16_t)(alignN / oneBlockElem);
                int64_t rowsDone = 0;
                while (rowsDone < fullRows) {
                    int64_t rowsLeft = fullRows - rowsDone;
                    uint8_t repeat = (uint8_t)(rowsLeft > SD_COPY_MAX_REPEAT ? SD_COPY_MAX_REPEAT : rowsLeft);
                    Copy<T, false>(dst[rowsDone * alignN], tmp[rowsDone * oneBlockElem], MASK_PLACEHOLDER, repeat,
                                   {1, 0, rowBlocks, 1});
                    rowsDone += repeat;
                }
                SetMaskNorm();
                ResetMask();
            }
            for (int64_t r = fullRows; r < M; r++) {
                T v = src.GetValue(r);
                Duplicate<T>(dst[r * alignN], v, (int32_t)alignN);
            }
        } else {
            // 标量 (1,1)->(M,alignN)：单点 GetValue + 填整块
            T v = src.GetValue(0);
            FillScalar(dst, v, M * alignN);
        }
    }

    template <int KIND1, int KIND2>
    __aicore__ inline void ComputeBrcInt64(const LocalTensor<T>& dst, const LocalTensor<T>& a, const LocalTensor<T>& b,
                                           int64_t M, int64_t N, int64_t alignN)
    {
        if (KIND1 == 0 && KIND2 == 0 && alignN == N) {
            int64_t total = M * N;
            int64_t i = 0;
            for (; i + 3 < total; i += 4) {
                int64_t d0 = a.GetValue(i) - b.GetValue(i);
                int64_t d1 = a.GetValue(i + 1) - b.GetValue(i + 1);
                int64_t d2 = a.GetValue(i + 2) - b.GetValue(i + 2);
                int64_t d3 = a.GetValue(i + 3) - b.GetValue(i + 3);
                dst.SetValue(i, d0 * d0);
                dst.SetValue(i + 1, d1 * d1);
                dst.SetValue(i + 2, d2 * d2);
                dst.SetValue(i + 3, d3 * d3);
            }
            for (; i < total; i++) {
                int64_t d = a.GetValue(i) - b.GetValue(i);
                dst.SetValue(i, d * d);
            }
            return;
        }
        if constexpr (KIND1 == 1 || KIND2 == 1) {
            if constexpr (KIND1 == 1 && KIND2 == 1) {
                for (int64_t c = 0; c < N; c++) {
                    int64_t va = a.GetValue(c);
                    int64_t vb = b.GetValue(c);
                    int64_t dd = (va - vb) * (va - vb);
                    for (int64_t r = 0; r < M; r++) {
                        dst.SetValue(r * alignN + c, dd);
                    }
                }
            } else if constexpr (KIND1 == 1) {
                for (int64_t c = 0; c < N; c++) {
                    int64_t va = a.GetValue(c);
                    int64_t r = 0;
                    for (; r + 3 < M; r += 4) {
                        int64_t d0 = va - ((KIND2 == 0) ? b.GetValue(r * alignN + c) :
                                           (KIND2 == 2) ? b.GetValue(r) :
                                                          b.GetValue(0));
                        int64_t d1 = va - ((KIND2 == 0) ? b.GetValue((r + 1) * alignN + c) :
                                           (KIND2 == 2) ? b.GetValue(r + 1) :
                                                          b.GetValue(0));
                        int64_t d2 = va - ((KIND2 == 0) ? b.GetValue((r + 2) * alignN + c) :
                                           (KIND2 == 2) ? b.GetValue(r + 2) :
                                                          b.GetValue(0));
                        int64_t d3 = va - ((KIND2 == 0) ? b.GetValue((r + 3) * alignN + c) :
                                           (KIND2 == 2) ? b.GetValue(r + 3) :
                                                          b.GetValue(0));
                        dst.SetValue(r * alignN + c, d0 * d0);
                        dst.SetValue((r + 1) * alignN + c, d1 * d1);
                        dst.SetValue((r + 2) * alignN + c, d2 * d2);
                        dst.SetValue((r + 3) * alignN + c, d3 * d3);
                    }
                    for (; r < M; r++) {
                        int64_t idx = r * alignN + c;
                        int64_t vb = (KIND2 == 0) ? b.GetValue(idx) : (KIND2 == 2) ? b.GetValue(r) : b.GetValue(0);
                        dst.SetValue(idx, (va - vb) * (va - vb));
                    }
                }
            } else {
                for (int64_t c = 0; c < N; c++) {
                    int64_t vb = b.GetValue(c);
                    int64_t r = 0;
                    for (; r + 3 < M; r += 4) {
                        int64_t d0 = ((KIND1 == 0) ? a.GetValue(r * alignN + c) :
                                      (KIND1 == 2) ? a.GetValue(r) :
                                                     a.GetValue(0)) -
                                     vb;
                        int64_t d1 = ((KIND1 == 0) ? a.GetValue((r + 1) * alignN + c) :
                                      (KIND1 == 2) ? a.GetValue(r + 1) :
                                                     a.GetValue(0)) -
                                     vb;
                        int64_t d2 = ((KIND1 == 0) ? a.GetValue((r + 2) * alignN + c) :
                                      (KIND1 == 2) ? a.GetValue(r + 2) :
                                                     a.GetValue(0)) -
                                     vb;
                        int64_t d3 = ((KIND1 == 0) ? a.GetValue((r + 3) * alignN + c) :
                                      (KIND1 == 2) ? a.GetValue(r + 3) :
                                                     a.GetValue(0)) -
                                     vb;
                        dst.SetValue(r * alignN + c, d0 * d0);
                        dst.SetValue((r + 1) * alignN + c, d1 * d1);
                        dst.SetValue((r + 2) * alignN + c, d2 * d2);
                        dst.SetValue((r + 3) * alignN + c, d3 * d3);
                    }
                    for (; r < M; r++) {
                        int64_t idx = r * alignN + c;
                        int64_t va = (KIND1 == 0) ? a.GetValue(idx) : (KIND1 == 2) ? a.GetValue(r) : a.GetValue(0);
                        dst.SetValue(idx, (va - vb) * (va - vb));
                    }
                }
            }
        } else if constexpr (KIND1 == 3 || KIND2 == 3) {
            int64_t vas = (KIND1 == 3) ? a.GetValue(0) : 0;
            int64_t vbs = (KIND2 == 3) ? b.GetValue(0) : 0;
            for (int64_t r = 0; r < M; r++) {
                int64_t base = r * alignN;
                int64_t var = (KIND1 == 2) ? a.GetValue(r) : vas;
                int64_t vbr = (KIND2 == 2) ? b.GetValue(r) : vbs;
                int64_t c = 0;
                for (; c + 3 < N; c += 4) {
                    int64_t va0 = (KIND1 == 0) ? a.GetValue(base + c) : var;
                    int64_t vb0 = (KIND2 == 0) ? b.GetValue(base + c) : vbr;
                    int64_t d0 = va0 - vb0;
                    int64_t va1 = (KIND1 == 0) ? a.GetValue(base + c + 1) : var;
                    int64_t vb1 = (KIND2 == 0) ? b.GetValue(base + c + 1) : vbr;
                    int64_t d1 = va1 - vb1;
                    int64_t va2 = (KIND1 == 0) ? a.GetValue(base + c + 2) : var;
                    int64_t vb2 = (KIND2 == 0) ? b.GetValue(base + c + 2) : vbr;
                    int64_t d2 = va2 - vb2;
                    int64_t va3 = (KIND1 == 0) ? a.GetValue(base + c + 3) : var;
                    int64_t vb3 = (KIND2 == 0) ? b.GetValue(base + c + 3) : vbr;
                    int64_t d3 = va3 - vb3;
                    dst.SetValue(base + c, d0 * d0);
                    dst.SetValue(base + c + 1, d1 * d1);
                    dst.SetValue(base + c + 2, d2 * d2);
                    dst.SetValue(base + c + 3, d3 * d3);
                }
                for (; c < N; c++) {
                    int64_t idx = base + c;
                    int64_t va = (KIND1 == 0) ? a.GetValue(idx) : var;
                    int64_t vb = (KIND2 == 0) ? b.GetValue(idx) : vbr;
                    int64_t d = va - vb;
                    dst.SetValue(idx, d * d);
                }
            }
        } else {
            for (int64_t r = 0; r < M; r++) {
                int64_t base = r * alignN;
                int64_t var = (KIND1 == 2) ? a.GetValue(r) : 0;
                int64_t vbr = (KIND2 == 2) ? b.GetValue(r) : 0;
                int64_t c = 0;
                for (; c + 3 < N; c += 4) {
                    int64_t va0 = (KIND1 == 0) ? a.GetValue(base + c) : var;
                    int64_t vb0 = (KIND2 == 0) ? b.GetValue(base + c) : vbr;
                    int64_t d0 = va0 - vb0;
                    int64_t va1 = (KIND1 == 0) ? a.GetValue(base + c + 1) : var;
                    int64_t vb1 = (KIND2 == 0) ? b.GetValue(base + c + 1) : vbr;
                    int64_t d1 = va1 - vb1;
                    int64_t va2 = (KIND1 == 0) ? a.GetValue(base + c + 2) : var;
                    int64_t vb2 = (KIND2 == 0) ? b.GetValue(base + c + 2) : vbr;
                    int64_t d2 = va2 - vb2;
                    int64_t va3 = (KIND1 == 0) ? a.GetValue(base + c + 3) : var;
                    int64_t vb3 = (KIND2 == 0) ? b.GetValue(base + c + 3) : vbr;
                    int64_t d3 = va3 - vb3;
                    dst.SetValue(base + c, d0 * d0);
                    dst.SetValue(base + c + 1, d1 * d1);
                    dst.SetValue(base + c + 2, d2 * d2);
                    dst.SetValue(base + c + 3, d3 * d3);
                }
                for (; c < N; c++) {
                    int64_t idx = base + c;
                    int64_t va = (KIND1 == 0) ? a.GetValue(idx) : var;
                    int64_t vb = (KIND2 == 0) ? b.GetValue(idx) : vbr;
                    int64_t d = va - vb;
                    dst.SetValue(idx, d * d);
                }
            }
        }
    }

    template <int KIND1>
    __aicore__ inline void ComputeBrcInt64Dispatch2(const LocalTensor<T>& dst, const LocalTensor<T>& a,
                                                    const LocalTensor<T>& b, int64_t M, int64_t N, int64_t alignN,
                                                    int kind2)
    {
        if (kind2 == 0)
            ComputeBrcInt64<KIND1, 0>(dst, a, b, M, N, alignN);
        else if (kind2 == 1)
            ComputeBrcInt64<KIND1, 1>(dst, a, b, M, N, alignN);
        else if (kind2 == 2)
            ComputeBrcInt64<KIND1, 2>(dst, a, b, M, N, alignN);
        else
            ComputeBrcInt64<KIND1, 3>(dst, a, b, M, N, alignN);
    }

    __aicore__ inline void ComputeBrcInt64Dispatch(const LocalTensor<T>& dst, const LocalTensor<T>& a,
                                                   const LocalTensor<T>& b, int64_t M, int64_t N, int64_t alignN,
                                                   int kind1, int kind2)
    {
        if (kind1 == 0)
            ComputeBrcInt64Dispatch2<0>(dst, a, b, M, N, alignN, kind2);
        else if (kind1 == 1)
            ComputeBrcInt64Dispatch2<1>(dst, a, b, M, N, alignN, kind2);
        else if (kind1 == 2)
            ComputeBrcInt64Dispatch2<2>(dst, a, b, M, N, alignN, kind2);
        else
            ComputeBrcInt64Dispatch2<3>(dst, a, b, M, N, alignN, kind2);
    }

    __aicore__ inline void AdvanceIdx(int64_t idx[], int64_t ubSplitAxis, int64_t ubOuter)
    {
        idx[ubSplitAxis]++;
        if (idx[ubSplitAxis] >= ubOuter) {
            idx[ubSplitAxis] = 0;
            for (int64_t i = ubSplitAxis - 1; i >= 0; i--) {
                idx[i]++;
                if (idx[i] < tiling_->outDims[i])
                    break;
                idx[i] = 0;
            }
        }
    }
};

#endif // SQUAREDDIFFERENCE_H
