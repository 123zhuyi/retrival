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
 * \file arg_max_with_value_nlast.h
 * \brief NLAST pattern (lastDim > 1): every output reduces axisSize elements strided by lastDim.
 *
 * For each output tile (up to innerTile columns of one firstDim row) we keep a running extremum value and its
 * winning index across the axis: each axis row is loaded as a contiguous plane (the lastDim GM stride
 * de-interleaves it), compares against the running extremum, and updates the index only on a *strict*
 * improvement (the compare mask is "not better", so ties keep the earlier axis row = first-occurrence).
 *
 * Two compute paths:
 *   ComputeNative - half input with axis <= 2048: reduce directly in fp16 (index is integer-exact there).
 *                   No cast, and the compare/select/extremum sequence runs 128 lanes/repeat instead of 64.
 *   ComputeCast   - everything else: reduce in fp32 so the int32 index stays exact (int16 is exact too).
 *
 * Small-lastDim batch path (host sets nlBf_ > 1): reduce nlBf_ consecutive firstDim planes together. Each axis
 * row loads the planes side by side, padded to nlIPad_, so one extremum combine covers every plane and amortizes
 * the scalar axis loop. Each plane's lastDim outputs are contiguous in GM and can be stored without compaction.
 *
 * Axis-split path (host sets splitAxis_==3): the default path already vectorizes across lastDim columns, so every
 * core retains all output columns while reducing a different axis slice. The distributed combine then folds the
 * partial values and indices. This path is scoped to nlBf_==1, firstDim==1, and one innerTile of output columns.
 */
#ifndef ARG_MAX_WITH_VALUE_NLAST_H
#define ARG_MAX_WITH_VALUE_NLAST_H

#include "arg_max_with_value_base.h"

namespace ArgWithValueNs {
using namespace AscendC;

template <typename T, bool IS_MIN, uint32_t SCHEDULE>
class ArgNLast : public ArgBase<T, IS_MIN> {
    static constexpr bool BATCH = SCHEDULE == ARG_SCH_NLAST_BATCH;
    static constexpr bool TREE = SCHEDULE == ARG_SCH_NLAST_TREE;
    static constexpr bool SPLIT = SCHEDULE == ARG_SCH_NLAST_SPLIT;

public:
    __aicore__ inline ArgNLast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR ws,
                                __tiling_data_ptr__ ArgMaxWithValueTilingData* t)
    {
        this->InitBase(x, indice, values, t);
        iLen_ = t->innerTile; // 128-aligned (host) so both fp16 (128) and fp32 (64) tiles fit
        aCap_ = t->axisTile;
        nlBf_ = t->nlBf;     // >1: batch nlBf_ consecutive firstDim planes per reduce
        nlIPad_ = t->nlIPad; // per-plane UB stride (lastDim rounded to 32B) for the batch path
        usedCoreNum_ = t->usedCoreNum;
        axisPerCore_ = t->axisPerCore;
        if (iLen_ == 0)
            iLen_ = 128;
        if (aCap_ == 0)
            aCap_ = 1;
        if (nlBf_ == 0)
            nlBf_ = 1;
        if (nlIPad_ == 0)
            nlIPad_ = iLen_;
        native_ = IsSameType<T, half>::value && this->axis_ <= NATIVE_AXIS_LIM;
        if constexpr (SPLIT) {
            sliceBase_ = GetBlockIdx() * axisPerCore_;
            uint32_t remain = this->axis_ - sliceBase_;
            sliceLen_ = (sliceBase_ >= this->axis_) ? 0 : (remain < axisPerCore_ ? remain : axisPerCore_);
            valStride2d_ = this->RoundUp(this->lastDim_, 8u);
        }

        // Widest tile any path needs: per-output (iLen_) or one batch group. 128-round the batch width so the
        // reduce's cn = RoundUp(bn*iPad, 64|128) never exceeds the running buffers.
        // Size the batch buffers for the planes THIS core actually reduces per group (perCore/lastDim),
        // NOT the theoretical nlBf. When a core owns few planes (e.g. 1) but nlBf is large, oversized
        // buffers shrink treeAc and explode the per-axis chunk count -> per-chunk load/sync scalar overhead
        // dominates. Right-sizing raises treeAc so the axis is reduced in far fewer chunks.
        uint32_t coreBn = (this->lastDim_ > 0u) ? (t->perCore / this->lastDim_) : nlBf_;
        if (coreBn < 1u)
            coreBn = 1u;
        if (coreBn > nlBf_)
            coreBn = nlBf_;
        uint32_t batchW = (BATCH || TREE) ? this->RoundUp(coreBn * nlIPad_, 128) : 0;
        uint32_t w = iLen_ > batchW ? iLen_ : batchW;                     // running-buffer / output width
        uint32_t loadW = aCap_ * iLen_ > batchW ? aCap_ * iLen_ : batchW; // input-tile width

        inputCap_ = loadW;
        outputCap_ = w;
        uint32_t ub = 0;
        inputSlotBytes_ = this->RoundUp(loadW * sizeof(T), 32u);
        inputAddr_ = Reserve(ub, 2u * inputSlotBytes_);
        outValueAddr_ = Reserve(ub, w * sizeof(T));
        outIndexAddr_ = Reserve(ub, w * sizeof(int32_t));
        if (!IsSameType<T, float>::value && !native_) // cast scratch only for the fp32-reduce path
            srcAddr_ = Reserve(ub, loadW * sizeof(float));
        // Running extremum plus two index ping-pong buffers, sized in fp32 (also holds fp16-native half views).
        curValAddr_ = Reserve(ub, w * sizeof(float));
        idxAAddr_ = Reserve(ub, w * sizeof(float));
        idxBAddr_ = Reserve(ub, w * sizeof(float));
        maskAddr_ = Reserve(ub, this->RoundUp(w, 256) / 8 + 32);
        treeAc_ = 0;
        if constexpr (TREE) {
            uint32_t bw = w;
            uint32_t ac = 100000u / (3u * bw * 4u + (IsSameType<T, float>::value ? 0u : bw * (uint32_t)sizeof(T)));
            uint32_t p = 2u;
            while (p * 2u <= ac && p < 256u)
                p *= 2u;
            treeAc_ = p;
            treeValAddr_ = Reserve(ub, treeAc_ * bw * sizeof(float));
            treeIdxAddr_ = Reserve(ub, treeAc_ * bw * sizeof(float));
            treeS1Addr_ = Reserve(ub, (treeAc_ / 2u) * bw * sizeof(float));
            treeS2Addr_ = Reserve(ub, (treeAc_ / 2u) * bw * sizeof(float));
            uint32_t mbytes = this->RoundUp((treeAc_ / 2u) * bw, 256) / 8 + 32;
            treeM0Addr_ = Reserve(ub, mbytes);
            mask2Addr_ = Reserve(ub, mbytes);
            if (!IsSameType<T, float>::value)
                treeRawAddr_ = Reserve(ub, treeAc_ * bw * sizeof(T));
        }
        if constexpr (SPLIT) {
            workspaceValGm_ = reinterpret_cast<__gm__ float*>(ws);
            workspaceArgGm_ = reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(ws) +
                                                                (uint64_t)usedCoreNum_ * valStride2d_ * sizeof(float));
            // CombineAxisSplit's fourth 32-bit scratch (accV/accI/candV reuse curValBuf_/idxABuf_/idxBBuf_,
            // sequenced after this core's own ComputeCast/Native + DrainAxisSplitToWs are done with them).
            combCandIAddr_ = Reserve(ub, this->RoundUp(this->lastDim_, 8u) * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (SPLIT) {
            ProcessAxisSplit();
        } else if constexpr (BATCH || TREE) {
            ProcessBatch();
        } else {
            uint32_t o = this->oStart_, oEnd = this->oStart_ + this->oLen_;
            while (o < oEnd) {
                uint32_t f = o / this->lastDim_;
                uint32_t l = o % this->lastDim_;
                uint32_t blockLen = this->lastDim_ - l;
                if (blockLen > oEnd - o)
                    blockLen = oEnd - o;
                for (uint32_t io = 0; io < blockLen; io += iLen_) {
                    uint32_t ilen = (blockLen - io) < iLen_ ? (blockLen - io) : iLen_;
                    if constexpr (IsSameType<T, half>::value) {
                        if (native_) {
                            ComputeNative(f, l + io, ilen);
                            Drain(o + io, ilen);
                            continue;
                        }
                    }
                    LocalTensor<int32_t> resultIdx = ComputeCast(f, l + io, ilen);
                    Drain(o + io, ilen, resultIdx);
                }
                o += blockLen;
            }
        }
    }

private:
    __aicore__ inline uint32_t Reserve(uint32_t& cursor, uint32_t bytes)
    {
        uint32_t addr = cursor;
        cursor = this->RoundUp(cursor + bytes, 32u);
        return addr;
    }

    template <typename U>
    __aicore__ inline LocalTensor<U> Ub(uint32_t addr)
    {
        return LocalTensor<U>(TPosition::VECCALC, addr, (192u * 1024u - addr) / sizeof(U));
    }

    __aicore__ inline LocalTensor<T> Input(uint32_t slot)
    {
        return LocalTensor<T>(TPosition::VECCALC, inputAddr_ + slot * inputSlotBytes_, inputCap_);
    }

    __aicore__ inline LocalTensor<T> OutValue()
    {
        return LocalTensor<T>(TPosition::VECCALC, outValueAddr_, outputCap_);
    }

    __aicore__ inline LocalTensor<int32_t> OutIndex()
    {
        return LocalTensor<int32_t>(TPosition::VECCALC, outIndexAddr_, outputCap_);
    }

    __aicore__ inline LocalTensor<float> Fp32Accumulator()
    {
        if constexpr (IsSameType<T, float>::value)
            return OutValue().template ReinterpretCast<float>();
        return Ub<float>(curValAddr_);
    }

    // ============================ shared helper (both paths below) ============================
    // GM base of axis row aStart for output (f, innerOff).
    __aicore__ inline uint64_t RowBase(uint32_t f, uint32_t innerOff, uint32_t aStart)
    {
        return (uint64_t)f * this->axis_ * this->lastDim_ + innerOff + (uint64_t)aStart * this->lastDim_;
    }

    // ============================ BATCH path (host sets nlBf_ > 1) ============================
    // reduce nlBf_ consecutive planes together: the per-output reduce below wastes most vector lanes and
    // is scalar-bound, so amortise the axis loop's scalar cost across nlBf_ planes at once (see file header).
    // The host aligns each core's output range to whole planes, so oStart_/oLen_ are multiples of lastDim_.
    __aicore__ inline void ProcessBatch()
    {
        uint32_t o = this->oStart_, oEnd = this->oStart_ + this->oLen_;
        while (o < oEnd) {
            uint32_t f0 = o / this->lastDim_;
            uint32_t planesLeft = (oEnd - o) / this->lastDim_;
            uint32_t bn = planesLeft < nlBf_ ? planesLeft : nlBf_;
            LocalTensor<int32_t> resultIdx = OutIndex();
            if constexpr (TREE) {
                if constexpr (IsSameType<T, half>::value) {
                    if (native_)
                        ComputeBatchTreeHalf(f0, bn); // axis<=2048: indices exact in half, no ToF cast
                    else
                        resultIdx = ComputeBatchTree(f0, bn);
                } else {
                    resultIdx = ComputeBatchTree(f0, bn);
                }
            } else {
                ComputeBatch(f0, bn);
            }
            DrainBatch(f0, bn, resultIdx);
            o += bn * this->lastDim_;
        }
    }

    // Reduce `bn` planes (f0 .. f0+bn-1), each lastDim_ columns padded to nlIPad_, over the strided axis.
    // One extremum combine per axis row covers all bn planes, amortizing the axis-loop scalar work across them.
    __aicore__ inline void ComputeBatch(uint32_t f0, uint32_t bn)
    {
        uint32_t w = bn * nlIPad_;
        uint64_t planeBase = (uint64_t)f0 * this->axis_ * this->lastDim_;
        // fp16-native (half, axis <= 2048): reduce directly in fp16. Guarded by if constexpr so the half-only
        // this->RawAdds() is never compiled for bf16/int16 (ComputeBatch is instantiated for every dtype).
        if constexpr (IsSameType<T, half>::value) {
            if (native_) {
                uint32_t cn = this->RoundUp(w, 128);
                LocalTensor<T> curV = OutValue();
                LocalTensor<T> idxA = Ub<T>(idxAAddr_);
                LocalTensor<T> idxB = Ub<T>(idxBAddr_);
                LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
                this->RawDupPad(curV, cn);
                this->RawDup(idxA, (T)0, cn);
                this->LoadRowsNoFill(Input(0), planeBase, bn, this->lastDim_, nlIPad_, this->axis_ * this->lastDim_);
                for (uint32_t a = 0; a < this->axis_; ++a) {
                    LocalTensor<T> row = Input(a & 1u);
                    this->template Sync<HardEvent::MTE2_V>();
                    if (a + 1 < this->axis_) { // prefetch next axis row under this row's compute
                        if (a > 0)
                            this->template Sync<HardEvent::V_MTE2>();
                        LocalTensor<T> xN = Input((a + 1u) & 1u);
                        this->LoadRowsNoFill(xN, planeBase + (uint64_t)(a + 1) * this->lastDim_, bn, this->lastDim_,
                                             nlIPad_, this->axis_ * this->lastDim_);
                    }
                    T aVal = (T)(int32_t)a;
                    this->RawCompare(mask, row, curV, IS_MIN ? CMPMODE::GE : CMPMODE::LE,
                                     cn); // "not better" -> keep idxA
                    this->RawSelectExtremum(idxB, mask, idxA, aVal, curV, curV, row, cn);
                    LocalTensor<T> tmp = idxA;
                    idxA = idxB;
                    idxB = tmp;
                }
                this->template Sync<HardEvent::V_MTE2>();
                LocalTensor<int32_t> oidx = OutIndex();
                this->template RawCast<RoundMode::CAST_RINT>(oidx, idxA, w);
                return;
            }
        }
        // fp32 reduce (bf16 / int16 / fp32, and half with axis > 2048): index stays integer-exact in fp32.
        uint32_t cn = this->RoundUp(w, 64);
        LocalTensor<float> curV = Fp32Accumulator();
        LocalTensor<float> idxA = Ub<float>(idxAAddr_);
        LocalTensor<float> idxB = Ub<float>(idxBAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        this->RawDupInitF(curV, cn);
        this->RawDup(idxA, 0.0f, cn);
        this->LoadRowsNoFill(Input(0), planeBase, bn, this->lastDim_, nlIPad_, this->axis_ * this->lastDim_);
        for (uint32_t a = 0; a < this->axis_; ++a) {
            LocalTensor<T> x = Input(a & 1u);
            this->template Sync<HardEvent::MTE2_V>();
            if (a + 1u < this->axis_) {
                if (a > 0u)
                    this->template Sync<HardEvent::V_MTE2>();
                this->LoadRowsNoFill(Input((a + 1u) & 1u), planeBase + (uint64_t)(a + 1u) * this->lastDim_, bn,
                                     this->lastDim_, nlIPad_, this->axis_ * this->lastDim_);
            }
            LocalTensor<float> row = this->ToF(x, Ub<float>(srcAddr_), w);
            float aVal = (float)(int32_t)a;
            this->RawCompare(mask, row, curV, IS_MIN ? CMPMODE::GE : CMPMODE::LE, cn);
            this->RawSelectExtremum(idxB, mask, idxA, aVal, curV, curV, row, cn);
            LocalTensor<float> tmp = idxA;
            idxA = idxB;
            idxB = tmp;
        }
        this->template Sync<HardEvent::V_MTE2>();
        LocalTensor<T> oval = OutValue();
        LocalTensor<int32_t> oidx = OutIndex();
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, curV, w);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, idxA, w);
    }

    // Large-axis fp32 batch reduce via chunked contiguous-half TREE (O(log axis) vec vs O(axis) scalar).
    // First-occurrence semantics are preserved by the six-operation Compare/Select/extremum combine.
    __aicore__ inline LocalTensor<int32_t> ComputeBatchTree(uint32_t f0, uint32_t bn)
    {
        uint32_t w = this->RoundUp(bn * nlIPad_, 128);
        uint32_t cnW = this->RoundUp(w, 64);
        uint64_t planeBase = (uint64_t)f0 * this->axis_ * this->lastDim_;
        LocalTensor<float> curV = Fp32Accumulator();
        LocalTensor<int32_t> curI = Ub<int32_t>(idxAAddr_);
        LocalTensor<float> tv = Ub<float>(treeValAddr_);
        LocalTensor<int32_t> ti = Ub<int32_t>(treeIdxAddr_);
        LocalTensor<float> s1 = Ub<float>(treeS1Addr_);
        LocalTensor<int32_t> s1I = Ub<int32_t>(treeS1Addr_);
        LocalTensor<int32_t> s2I = Ub<int32_t>(treeS2Addr_);
        LocalTensor<uint8_t> m0 = Ub<uint8_t>(treeM0Addr_);
        LocalTensor<uint8_t> m1 = Ub<uint8_t>(mask2Addr_);
        this->RawDupInitF(curV, cnW);
        this->RawDup(curI, (int32_t)0, cnW);
        for (uint32_t aStart = 0; aStart < this->axis_; aStart += treeAc_) {
            uint32_t aN = (this->axis_ - aStart) < treeAc_ ? (this->axis_ - aStart) : treeAc_;
            if (aN < treeAc_)
                this->RawDup(ti[aN * w], (int32_t)this->axis_, (treeAc_ - aN) * w);
            if constexpr (IsSameType<T, float>::value) {
                this->RawDupInitF(tv, treeAc_ * w);
                this->template Sync<HardEvent::V_MTE2>();
                for (uint32_t a = 0; a < aN; ++a) {
                    this->LoadRowsNoFill(tv[a * w], planeBase + (uint64_t)(aStart + a) * this->lastDim_, bn,
                                         this->lastDim_, nlIPad_, this->axis_ * this->lastDim_);
                    this->RawDup(ti[a * w], (int32_t)(aStart + a), w);
                }
                this->template Sync<HardEvent::MTE2_V>();
            } else {
                LocalTensor<T> raw = Ub<T>(treeRawAddr_);
                this->RawDupPad(raw, treeAc_ * w);
                this->template Sync<HardEvent::V_MTE2>();
                for (uint32_t a = 0; a < aN; ++a) {
                    this->LoadRowsNoFill(raw[a * w], planeBase + (uint64_t)(aStart + a) * this->lastDim_, bn,
                                         this->lastDim_, nlIPad_, this->axis_ * this->lastDim_);
                    this->RawDup(ti[a * w], (int32_t)(aStart + a), w);
                }
                this->template Sync<HardEvent::MTE2_V>();
                this->ToF(raw, Ub<float>(treeValAddr_), treeAc_ * w);
            }
            uint32_t n = treeAc_;
            while (n > 1) {
                uint32_t half = n / 2;
                uint32_t cnt = half * w;
                this->template RawCompareSelect<IS_MIN ? CMPMODE::LE : CMPMODE::GE>(
                    s1I.template ReinterpretCast<float>(), tv, tv[half * w], ti.template ReinterpretCast<float>(),
                    ti[half * w].template ReinterpretCast<float>(), cnt);
                this->template RawCompareSelect<IS_MIN ? CMPMODE::LE : CMPMODE::GE>(
                    s2I.template ReinterpretCast<float>(), tv[half * w], tv,
                    ti[half * w].template ReinterpretCast<float>(), ti.template ReinterpretCast<float>(), cnt);
                this->RawMin(ti, s1I, s2I, cnt);
                if constexpr (IS_MIN)
                    this->RawMin(tv, tv, tv[half * w], cnt);
                else
                    this->RawMax(tv, tv, tv[half * w], cnt);
                n = half;
            }
            if constexpr (IS_MIN)
                this->RawMin(s1, curV, tv, cnW);
            else
                this->RawMax(s1, curV, tv, cnW);
            this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(
                s2I.template ReinterpretCast<float>(), tv, curV, ti.template ReinterpretCast<float>(),
                curI.template ReinterpretCast<float>(), cnW);
            this->RawMove(curV, s1, cnW);
            this->RawMove(curI, s2I, cnW);
            pipe_barrier(PIPE_V);
        }
        LocalTensor<T> oval = OutValue();
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, curV, w);
        return curI;
    }

    // Native-half tree (fp16, axis<=2048): same butterfly as ComputeBatchTree but values AND indices stay in
    // half -- no fp16->fp32 ToF cast and half the vector lanes per op, attacking the vec-bound critical path.
    // Indices (integers <= axis <= 2048) are exact in half; extremum selection only compares the values.
    __aicore__ inline void ComputeBatchTreeHalf(uint32_t f0, uint32_t bn)
    {
        if constexpr (IsSameType<T, half>::value) {
            uint32_t w = this->RoundUp(bn * nlIPad_, 128);
            uint32_t cnW = this->RoundUp(w, 128);
            uint64_t planeBase = (uint64_t)f0 * this->axis_ * this->lastDim_;
            LocalTensor<half> curV = OutValue().template ReinterpretCast<half>();
            LocalTensor<half> curI = Ub<half>(idxAAddr_);
            LocalTensor<half> tv = Ub<half>(treeValAddr_);
            LocalTensor<half> ti = Ub<half>(treeIdxAddr_);
            LocalTensor<half> s1 = Ub<half>(treeS1Addr_);
            LocalTensor<half> s2 = Ub<half>(treeS2Addr_);
            LocalTensor<uint8_t> m0 = Ub<uint8_t>(treeM0Addr_);
            LocalTensor<uint8_t> m1 = Ub<uint8_t>(mask2Addr_);
            this->RawDupPad(curV, cnW);
            this->RawDup(curI, (half)0, cnW);
            for (uint32_t aStart = 0; aStart < this->axis_; aStart += treeAc_) {
                uint32_t aN = (this->axis_ - aStart) < treeAc_ ? (this->axis_ - aStart) : treeAc_;
                if (aN < treeAc_)
                    this->RawDup(ti[aN * w], (half)(int32_t)this->axis_, (treeAc_ - aN) * w);
                this->RawDupPad(tv, treeAc_ * w);
                for (uint32_t a = 0; a < aN; ++a)
                    this->RawDup(ti[a * w], (half)(int32_t)(aStart + a), w);
                this->template Sync<HardEvent::V_MTE2>();
                for (uint32_t a = 0; a < aN; ++a)
                    this->LoadRowsNoFill(tv[a * w], planeBase + (uint64_t)(aStart + a) * this->lastDim_, bn,
                                         this->lastDim_, nlIPad_, this->axis_ * this->lastDim_);
                this->template Sync<HardEvent::MTE2_V>();
                uint32_t n = treeAc_;
                while (n > 1) {
                    uint32_t hf = n / 2;
                    uint32_t cnt = hf * w;
                    this->template RawCompareSelect<IS_MIN ? CMPMODE::LE : CMPMODE::GE>(s1, tv, tv[hf * w], ti,
                                                                                        ti[hf * w], cnt);
                    this->template RawCompareSelect<IS_MIN ? CMPMODE::LE : CMPMODE::GE>(s2, tv[hf * w], tv, ti[hf * w],
                                                                                        ti, cnt);
                    this->RawMin(ti, s1, s2, cnt);
                    if constexpr (IS_MIN)
                        this->RawMin(tv, tv, tv[hf * w], cnt);
                    else
                        this->RawMax(tv, tv, tv[hf * w], cnt);
                    n = hf;
                }
                if constexpr (IS_MIN)
                    this->RawMin(s1, curV, tv, cnW);
                else
                    this->RawMax(s1, curV, tv, cnW);
                this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(s2, tv, curV, ti, curI, cnW);
                this->RawMove(curV, s1, cnW);
                this->RawMove(curI, s2, cnW);
                pipe_barrier(PIPE_V);
            }
            LocalTensor<int32_t> oidx = OutIndex();
            this->template RawCast<RoundMode::CAST_RINT>(oidx, curI, w);
        }
    }

    // Store each of the bn planes' lastDim_ output columns (contiguous in GM) from its nlIPad_-strided slice.
    __aicore__ inline void DrainBatch(uint32_t f0, uint32_t bn, const LocalTensor<int32_t>& resultIdx)
    {
        LocalTensor<T> oval = OutValue();
        this->template Sync<HardEvent::V_MTE3>();
        constexpr uint32_t valueBlock = 32u / sizeof(T);
        const uint32_t valueSrcGap = (nlIPad_ - this->RoundUp(this->lastDim_, valueBlock)) / valueBlock;
        const uint32_t indexSrcGap = (nlIPad_ - this->RoundUp(this->lastDim_, 8u)) / 8u;
        __ubuf__ T* valueUb = reinterpret_cast<__ubuf__ T*>(oval.GetPhyAddr());
        __ubuf__ int32_t* indexUb = reinterpret_cast<__ubuf__ int32_t*>(resultIdx.GetPhyAddr());
        __gm__ T* valueGm = this->valuesGm_ + (uint64_t)f0 * this->lastDim_;
        __gm__ int32_t* indexGm = this->indiceGm_ + (uint64_t)f0 * this->lastDim_;
        if constexpr (sizeof(T) == 2u)
            copy_ubuf_to_gm_align_b16(valueGm, valueUb, 0, static_cast<uint16_t>(bn), this->lastDim_ * sizeof(T), 0, 0,
                                      valueSrcGap, 0);
        else
            copy_ubuf_to_gm_align_b32(valueGm, valueUb, 0, static_cast<uint16_t>(bn), this->lastDim_ * sizeof(T), 0, 0,
                                      valueSrcGap, 0);
        copy_ubuf_to_gm_align_b32(indexGm, indexUb, 0, static_cast<uint16_t>(bn), this->lastDim_ * sizeof(int32_t), 0,
                                  0, indexSrcGap, 0);
        this->template SyncMte3Complete<HardEvent::MTE3_V>();
    }

    // ============================ PER-OUTPUT path (nlBf_ == 1, the default) ============================
    // fp32-value reduce path (bf16 / int16, and half with axis > 2048). aStart0/aLenArg default to the full axis
    // (aLenArg==0 -> this->axis_). Indices stay int32 throughout; vsel bit-moves them through an fp32 lane view,
    // so every int32 axis position remains exact without adding a conversion instruction.
    __aicore__ inline LocalTensor<int32_t> ComputeCast(uint32_t f, uint32_t innerOff, uint32_t ilen,
                                                       uint32_t aStart0 = 0, uint32_t aLenArg = 0)
    {
        uint32_t aTotal = aLenArg ? aLenArg : this->axis_;
        uint32_t aEnd = aStart0 + aTotal;
        uint32_t cn = this->RoundUp(ilen, 64); // fp32 compare/select lanes are 64-aligned
        LocalTensor<float> curV = Fp32Accumulator();
        LocalTensor<int32_t> idxA = Ub<int32_t>(idxAAddr_);
        LocalTensor<int32_t> idxB = Ub<int32_t>(idxBAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        this->RawDupInitF(curV, cn); // Initialize the running extremum with the inactive sentinel.
        this->RawDup(idxA, (int32_t)0, cn);
        uint32_t aN0 = aTotal < aCap_ ? aTotal : aCap_;
        this->LoadRowsNoFill(Input(0), RowBase(f, innerOff, aStart0), aN0, ilen, iLen_, this->lastDim_);
        uint32_t block = 0;
        for (uint32_t aStart = aStart0; aStart < aEnd; aStart += aCap_) {
            uint32_t aN = (aEnd - aStart) < aCap_ ? (aEnd - aStart) : aCap_;
            LocalTensor<T> x = Input(block & 1u);
            this->template Sync<HardEvent::MTE2_V>();
            uint32_t aStartN = aStart + aCap_;
            if (aStartN < aEnd) { // prefetch next block under this block's compute
                uint32_t aNn = (aEnd - aStartN) < aCap_ ? (aEnd - aStartN) : aCap_;
                if (block > 0)
                    this->template Sync<HardEvent::V_MTE2>();
                LocalTensor<T> xN = Input((block + 1u) & 1u);
                this->LoadRowsNoFill(xN, RowBase(f, innerOff, aStartN), aNn, ilen, iLen_, this->lastDim_);
            }
            LocalTensor<float> blockF = this->ToF(x, Ub<float>(srcAddr_), aN * iLen_);
            for (uint32_t a = 0; a < aN; ++a) {
                LocalTensor<float> row = blockF[a * iLen_];
                int32_t aVal = (int32_t)(aStart + a);
                this->RawCompare(mask, row, curV, IS_MIN ? CMPMODE::GE : CMPMODE::LE, cn); // "not better" -> keep idxA
                this->RawSelectExtremum(idxB, mask, idxA, aVal, curV, curV, row, cn);
                LocalTensor<int32_t> tmp = idxA;
                idxA = idxB;
                idxB = tmp;
            }
            ++block;
        }
        this->template Sync<HardEvent::V_MTE2>();
        LocalTensor<T> oval = OutValue();
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, curV, ilen);
        return idxA;
    }

    // fp16-native path (half, axis <= 2048): reduce directly in fp16, index integer-exact. aStart0/aLenArg: see
    // ComputeCast's comment above (same optional-range convention).
    __aicore__ inline void ComputeNative(uint32_t f, uint32_t innerOff, uint32_t ilen, uint32_t aStart0 = 0,
                                         uint32_t aLenArg = 0)
    {
        uint32_t aTotal = aLenArg ? aLenArg : this->axis_;
        uint32_t aEnd = aStart0 + aTotal;
        uint32_t cn = this->RoundUp(ilen, 128); // fp16 compare/select lanes are 128-aligned
        LocalTensor<T> curV = OutValue();
        LocalTensor<T> idxA = Ub<T>(idxAAddr_);
        LocalTensor<T> idxB = Ub<T>(idxBAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        this->RawDupPad(curV, cn); // Initialize with a sentinel worse than any real value.
        this->RawDup(idxA, (T)0, cn);
        uint32_t aN0 = aTotal < aCap_ ? aTotal : aCap_;
        this->LoadRowsNoFill(Input(0), RowBase(f, innerOff, aStart0), aN0, ilen, iLen_, this->lastDim_);
        uint32_t block = 0;
        for (uint32_t aStart = aStart0; aStart < aEnd; aStart += aCap_) {
            uint32_t aN = (aEnd - aStart) < aCap_ ? (aEnd - aStart) : aCap_;
            LocalTensor<T> x = Input(block & 1u);
            this->template Sync<HardEvent::MTE2_V>();
            uint32_t aStartN = aStart + aCap_;
            if (aStartN < aEnd) { // prefetch next block
                uint32_t aNn = (aEnd - aStartN) < aCap_ ? (aEnd - aStartN) : aCap_;
                if (block > 0)
                    this->template Sync<HardEvent::V_MTE2>();
                LocalTensor<T> xN = Input((block + 1u) & 1u);
                this->LoadRowsNoFill(xN, RowBase(f, innerOff, aStartN), aNn, ilen, iLen_, this->lastDim_);
            }
            for (uint32_t a = 0; a < aN; ++a) {
                LocalTensor<T> row = x[a * iLen_];
                T aVal = (T)(int32_t)(aStart + a);
                this->RawCompare(mask, row, curV, IS_MIN ? CMPMODE::GE : CMPMODE::LE, cn);
                this->RawSelectExtremum(idxB, mask, idxA, aVal, curV, curV, row, cn);
                LocalTensor<T> tmp = idxA;
                idxA = idxB;
                idxB = tmp;
            }
            ++block;
        }
        this->template Sync<HardEvent::V_MTE2>();
        LocalTensor<int32_t> oidx = OutIndex();
        this->template RawCast<RoundMode::CAST_RINT>(oidx, idxA, ilen);
    }

    // ============================ shared output drain (per-output/batch paths above) ============================
    __aicore__ inline void Drain(uint32_t outOff, uint32_t n) { Drain(outOff, n, OutIndex()); }

    __aicore__ inline void Drain(uint32_t outOff, uint32_t n, const LocalTensor<int32_t>& oidx)
    {
        LocalTensor<T> oval = OutValue();
        this->template Sync<HardEvent::V_MTE3>();
        this->StoreOut(outOff, n, oval, oidx);
        this->template SyncMte3Complete<HardEvent::MTE3_V>();
    }

    // ============================ AXIS-SPLIT path (host sets splitAxis_==3) ============================
    // Every core reduces its own axis slice [sliceBase_, sliceBase_+sliceLen_) for ALL lastDim_ columns at once
    // (firstDim==1 is host-gated, so f=0 always and RowBase's f*axis_*lastDim_ term is always 0 regardless of
    // how the axis loop bound is sliced -- see ComputeCast/ComputeNative's aStart0/aLenArg params), publishes
    // (value, GLOBAL axis index) per column to its own workspace row, then a cross-core strict-first combine
    // (distributed across `used` cores, not serial on core 0) folds the used_ slices per column. Mirrors
    // ArgLast's proven ProcessSplitMulti/DrainToWs/CombineMulti (splitAxis==2) almost line for line.
    __aicore__ inline void ProcessAxisSplit()
    {
        uint32_t core = GetBlockIdx();
        uint32_t ilen = this->lastDim_; // == outSize_ here (firstDim==1); fits one iLen_-wide tile (host-gated small)
        LocalTensor<int32_t> resultIdx = OutIndex();
        if (sliceLen_ == 0) {
            // Defensive only: used = ceil(axisSize/axisPerCore_) by construction, so every launched core
            // (blockIdx in [0,used)) has sliceLen_>0. Never expected to trigger; publish a sentinel if it did.
            LocalTensor<T> oval = OutValue();
            LocalTensor<int32_t> oidx = OutIndex();
            this->RawDupPad(oval, ilen);
            this->RawDup(oidx, (int32_t)0, ilen);
        } else if constexpr (IsSameType<T, half>::value) {
            if (native_)
                ComputeNative(0, 0, ilen, sliceBase_, sliceLen_);
            else
                resultIdx = ComputeCast(0, 0, ilen, sliceBase_, sliceLen_);
        } else {
            resultIdx = ComputeCast(0, 0, ilen, sliceBase_, sliceLen_);
        }
        DrainAxisSplitToWs(core, ilen, resultIdx);
        // This core's MTE3 publish must complete before any combine read (MTE2) -- DMA GM access has no
        // cache-coherence issue, only pipe ordering; SyncAll then guarantees every core's writes land before reads
        // (same reasoning ArgLast::ProcessSplitMulti documents above its own SyncAll() call).
        this->template SyncMte3Complete<HardEvent::MTE3_MTE2>();
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
        CombineAxisSplit(core);
    }

    // Publish this core's value and exact int32 global index for all columns.
    __aicore__ inline void DrainAxisSplitToWs(uint32_t core, uint32_t n, const LocalTensor<int32_t>& resultIdx)
    {
        LocalTensor<T> oval = OutValue();
        LocalTensor<float> vF = oval.template ReinterpretCast<float>();
        if constexpr (!IsSameType<T, float>::value) {
            vF = Ub<float>(curValAddr_);
            this->template RawCast<RoundMode::CAST_NONE>(vF, oval, n);
        }
        this->template Sync<HardEvent::V_MTE3>();
        this->RawStore(workspaceValGm_ + (uint64_t)core * valStride2d_, vF, n * sizeof(float));
        this->RawStore(workspaceArgGm_ + (uint64_t)core * valStride2d_, resultIdx, n * sizeof(int32_t));
    }

    // this core folds the output columns [cb, cb+cr) it owns: read all usedCoreNum_ slices (strided) and keep
    // the strict-first (earliest slice = smallest global axis index) winner per column.
    __aicore__ inline void CombineAxisSplit(uint32_t core)
    {
        uint32_t cr0 = this->RoundUp((this->lastDim_ + usedCoreNum_ - 1) / usedCoreNum_, 8u);
        uint32_t cb = core * cr0;
        if (cb >= this->lastDim_)
            return;
        uint32_t cr = (this->lastDim_ - cb) < cr0 ? (this->lastDim_ - cb) : cr0;
        LocalTensor<float> accV = Fp32Accumulator();
        LocalTensor<int32_t> accI = Ub<int32_t>(idxAAddr_);
        LocalTensor<float> candV = Ub<float>(idxBAddr_);
        LocalTensor<int32_t> candI = Ub<int32_t>(combCandIAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        uint32_t cmpR = this->RoundUp(cr, 64);
        this->RawLoad(accV, workspaceValGm_ + cb, cr * sizeof(float));
        this->RawLoad(accI, workspaceArgGm_ + cb, cr * sizeof(int32_t));
        this->template Sync<HardEvent::MTE2_V>();
        for (uint32_t s = 1; s < usedCoreNum_; ++s) {
            this->RawLoad(candV, workspaceValGm_ + (uint64_t)s * valStride2d_ + cb, cr * sizeof(float));
            this->RawLoad(candI, workspaceArgGm_ + (uint64_t)s * valStride2d_ + cb, cr * sizeof(float));
            this->template Sync<HardEvent::MTE2_V>();
            this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(
                accI.template ReinterpretCast<float>(), candV, accV, candI.template ReinterpretCast<float>(),
                accI.template ReinterpretCast<float>(), cmpR);
            if constexpr (IS_MIN)
                this->RawMin(accV, accV, candV, (int32_t)cr);
            else
                this->RawMax(accV, accV, candV, (int32_t)cr);
            this->template Sync<HardEvent::V_MTE2>(); // reads done before the next slice overwrites candV/candI
        }
        LocalTensor<T> oval = OutValue();
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, accV, cr);
        Drain(cb, cr, accI);
    }

    constexpr static uint32_t TREE_AXIS_MIN = 64;
    constexpr static uint32_t NATIVE_AXIS_LIM = 2048; // fp16 represents integer indices exactly up to 2048

    uint32_t inputAddr_ = 0, inputSlotBytes_ = 0, inputCap_ = 0;
    uint32_t outValueAddr_ = 0, outIndexAddr_ = 0, outputCap_ = 0;
    uint32_t srcAddr_ = 0, curValAddr_ = 0, idxAAddr_ = 0, idxBAddr_ = 0, maskAddr_ = 0;
    uint32_t treeValAddr_ = 0, treeIdxAddr_ = 0, treeS1Addr_ = 0, treeS2Addr_ = 0;
    uint32_t treeM0Addr_ = 0, mask2Addr_ = 0, treeRawAddr_ = 0, combCandIAddr_ = 0;
    __gm__ float* workspaceValGm_;   // axis-split: [usedCoreNum][RoundUp(lastDim_,8)] value floats
    __gm__ int32_t* workspaceArgGm_; // axis-split: same-shaped exact GLOBAL axis indices
    uint32_t treeAc_ = 0;
    uint32_t iLen_ = 128, aCap_ = 1;
    uint32_t nlBf_ = 1, nlIPad_ = 128;
    uint32_t usedCoreNum_ = 1;
    uint32_t axisPerCore_ = 0, sliceBase_ = 0, sliceLen_ = 0, valStride2d_ = 0;
    bool native_ = false;
};
} // namespace ArgWithValueNs
#endif // ARG_MAX_WITH_VALUE_NLAST_H
