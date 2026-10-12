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
 * \file arg_max_with_value_last.h
 * \brief LAST pattern (lastDim == 1): every output reduces a contiguous run of axisSize elements.
 *
 * The axis width drives three reduce strategies (all reduce in fp32, all first-occurrence):
 *   axis <= 64   : ReduceTiny  - one WholeReduce per row tile (row width de-padded to 32B), gather straight out.
 *   65..512(16b) : ReduceSeg   - reduce each 64-chunk then fold with a strict compare (avoids the nc-multiple-of-8
 *                                strided level-2, so the row stays narrow, ~RoundUp(axis,64)).
 *   wider axes   : ReducePiece - one batched WholeReduce over all R*nc 64-chunks, then a strided level-2 reduce
 *                                value plus an index reconstruction that preserves the earliest winning chunk.
 * Axes beyond PIECE_AXIS are cut into pieces and folded with the same strict (first-occurrence) compare.
 */
#ifndef ARG_MAX_WITH_VALUE_LAST_H
#define ARG_MAX_WITH_VALUE_LAST_H

#include "arg_max_with_value_base.h"

namespace ArgWithValueNs {
using namespace AscendC;

template <typename T, bool IS_MIN, uint32_t SCHEDULE, bool GATHER>
class ArgLast : public ArgBase<T, IS_MIN> {
    static constexpr bool SPLIT1 = SCHEDULE == ARG_SCH_LAST_SPLIT1;
    static constexpr bool SPLIT2 = SCHEDULE == ARG_SCH_LAST_SPLIT2;
    static constexpr bool TINY = SCHEDULE == ARG_SCH_LAST_TINY;
    static constexpr bool PACK2 = SCHEDULE == ARG_SCH_LAST_PACK2;
    static constexpr bool PACK3 = SCHEDULE == ARG_SCH_LAST_PACK3;
    static constexpr bool PACK4 = SCHEDULE == ARG_SCH_LAST_PACK4;
    static constexpr bool PACK5 = SCHEDULE == ARG_SCH_LAST_PACK5;
    static constexpr bool PACKN = SCHEDULE == ARG_SCH_LAST_PACKN;
    static constexpr bool SEG = SCHEDULE == ARG_SCH_LAST_SEG;
    static constexpr bool SMALL = TINY || PACK2 || PACK3 || PACK4 || PACK5 || PACKN;

public:
    __aicore__ inline ArgLast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR ws,
                                __tiling_data_ptr__ ArgMaxWithValueTilingData* t)
    {
        this->InitBase(x, indice, values, t);
        if constexpr (GATHER) {
            const uint32_t core = GetBlockIdx();
            const uint32_t base = t->perCore;
            const uint32_t extra = t->bigCores;
            this->oStart_ = core < extra ? core * (base + 1u) : extra * (base + 1u) + (core - extra) * base;
            this->oLen_ = base + (core < extra ? 1u : 0u);
            if (this->oStart_ + this->oLen_ > this->outSize_)
                this->oLen_ = this->outSize_ - this->oStart_;
        }
        R_ = t->rowTile;
        if (R_ == 0)
            R_ = 1;
        axisPerCore_ = t->axisPerCore;
        usedCoreNum_ = t->usedCoreNum;
        gmStride_ = this->axis_; // GM row stride is always the FULL axis; a 2D slice reads a window of each row
        if constexpr (SPLIT2) {  // this core reduces only the slice [sliceBase_, sliceBase_+axisPerCore_) of each row
            sliceBase_ = GetBlockIdx() * axisPerCore_;
            this->axis_ = (sliceBase_ < gmStride_) ? MinU(axisPerCore_, gmStride_ - sliceBase_) : 1;
        }
        uint32_t axis = this->axis_;
        constexpr uint32_t BLK = 32 /
                                 sizeof(T); // elements per 32B (load alignment); also a multiple of 8 (fp32 reduce)
        // Hybrid: axis<=64 uses batched WholeReduce per row tile; 65..256 uses a row-wise segment reduction.
        if constexpr (SPLIT2) {
            const uint32_t tinyLim = IsSameType<T, half>::value ? 128u : 64u;
            splitTiny_ = axis <= tinyLim;
            splitSeg_ = axis > tinyLim && axis <= 256u;
            splitMicro_ = axis == 2u;
        }
        W_ = IsSmall() ? (IsPacked() ? axis : this->RoundUp(axis, BLK)) :
                         (IsSeg() ? this->RoundUp(axis, BLK) : RoundUp512(MinU(axis, PIECE_AXIS)));
        if constexpr (TINY)
            loadGroups_ = BLK / GcdU(gmStride_, BLK);
        nc_ = IsSmall() ? 1 : (IsSeg() ? (W_ + 63u) / 64u : W_ / 64u);
        R8_ = this->RoundUp(R_, 64);
        // Native-half whenever the value can reduce 128/chunk (W_ a multiple of 1024 -> nc128=W_/128 a multiple of 8,
        // 32B-block-aligned level-2). Reducing in half avoids the whole-input fp32 ToF cast -> drops srcBuf -> the host
        // can hand a larger row tile R and increase occupancy.
        useHalf_ = IsSameType<T, half>::value && !IsSmall() && !IsSeg() && (W_ % 1024u == 0u);
        uint32_t lastP = (axis % PIECE_AXIS == 0u) ? PIECE_AXIS : (axis % PIECE_AXIS);
        uint32_t finalSliceP = lastP;
        if constexpr (SPLIT2) {
            uint64_t finalSliceBase = (uint64_t)(usedCoreNum_ - 1u) * axisPerCore_;
            uint32_t finalSliceAxis = (uint32_t)(gmStride_ - finalSliceBase);
            finalSliceP = (finalSliceAxis % PIECE_AXIS == 0u) ? PIECE_AXIS : (finalSliceAxis % PIECE_AXIS);
        }
        noSrcBuf_ = useHalf_ && (RoundUp512(lastP) % 1024u == 0u) &&
                    (RoundUp512(finalSliceP) % 1024u == 0u); // every split slice must stay native-half

        // Packed micro paths load and de-interleave only axis_*R_ elements. Reserve one vector margin instead of
        // R_*W_ so the remaining UB can accommodate larger row tiles.
        uint32_t loadElems = IsPacked() ? (this->RoundUp(this->axis_ * R_, 128u) + 128u) : (R_ * W_);
        inputCap_ = loadElems;
        outputCap_ = R8_;
        uint32_t ub = 0;
        inputSlotBytes_ = this->RoundUp(loadElems * sizeof(T), 32u);
        outValueSlotBytes_ = this->RoundUp(R8_ * sizeof(T), 32u);
        outIndexSlotBytes_ = this->RoundUp(R8_ * sizeof(int32_t), 32u);
        inputAddr_ = Reserve(ub, 2u * inputSlotBytes_);
        outValueAddr_ = Reserve(ub, 2u * outValueSlotBytes_);
        outIndexAddr_ = Reserve(ub, 2u * outIndexSlotBytes_);
        // 64-align the chunk scratch: Compare/Select/Broadcast run over RoundUp(R*nc,64) lanes, so under-sizing
        // these would overflow into the neighbouring buffer (corrupting the value result for non-64-multiple R*nc).
        uint32_t rnc = this->RoundUp(R_ * nc_, 64);
        if constexpr (!IsSameType<T, float>::value)
            if (!noSrcBuf_ || SPLIT1) // split1 ProcessSplit ToF needs srcBuf
                srcAddr_ = Reserve(ub, loadElems * sizeof(float));
        redAddr_ = Reserve(ub, 2 * rnc * sizeof(float));
        cminAddr_ = Reserve(ub, rnc * sizeof(float));
        cidxAddr_ = Reserve(ub, rnc * sizeof(float));
        maskAddr_ = Reserve(ub, this->RoundUp(rnc, 256) / 8 + 32);
        if constexpr (PACK2) { // constant 1.0 vector for the axis==2 first-occurrence index select
            oneAddr_ = Reserve(ub, R8_ * sizeof(float));
            this->RawDup(Ub<float>(oneAddr_), 1.0f, R8_);
        }
        if constexpr (PACK3 || PACK4 || PACK5 || PACKN) {
            col2Addr_ = Reserve(ub, rnc * sizeof(float));
            col3Addr_ = Reserve(ub, rnc * sizeof(float));
            tmpAAddr_ = Reserve(ub, 2 * rnc * sizeof(float));
            tmpBAddr_ = Reserve(ub, 2 * rnc * sizeof(float));
            const0Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const0Addr_), 0.0f, rnc);
            const1Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const1Addr_), 1.0f, rnc);
            const2Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const2Addr_), 2.0f, rnc);
            const3Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const3Addr_), 3.0f, rnc);
        }
        if constexpr (PACK3 || PACK5) {
            uint32_t Ncol = this->axis_;
            uint32_t NR = Ncol * R_;
            uint32_t NR64 = this->RoundUp(NR, 64u);
            if constexpr (PACK5) {
                col4Addr_ = Reserve(ub, rnc * sizeof(float));
                const4Addr_ = Reserve(ub, rnc * sizeof(float));
                this->RawDup(Ub<float>(const4Addr_), 4.0f, rnc);
                pat3Addr_ = Reserve(ub, this->RoundUp(NR, 256) / 8 + 32);
                pat4Addr_ = Reserve(ub, this->RoundUp(NR, 256) / 8 + 32);
            }
            genAddr_ = Reserve(ub, NR64 * sizeof(float));
            gen2Addr_ = Reserve(ub, NR64 * sizeof(float));
            pat0Addr_ = Reserve(ub, this->RoundUp(NR, 256) / 8 + 32);
            pat1Addr_ = Reserve(ub, this->RoundUp(NR, 256) / 8 + 32);
            pat2Addr_ = Reserve(ub, this->RoundUp(NR, 256) / 8 + 32);
            LocalTensor<float> idxv = Ub<float>(genAddr_);
            LocalTensor<float> g = Ub<float>(gen2Addr_);
            this->RawIota(idxv, 0.0f, NR64);
            this->RawAdds(g, idxv, 0.5f, NR64);
            this->RawMuls(g, g, 1.0f / (float)(int32_t)Ncol, NR64);
            this->template RawCast<RoundMode::CAST_FLOOR>(g.template ReinterpretCast<int32_t>(), g,
                                                          NR64); // floor((idx+0.5)/N)
            this->template RawCast<RoundMode::CAST_NONE>(g, g.template ReinterpretCast<int32_t>(), NR64);
            this->RawMuls(g, g, (float)(int32_t)Ncol, NR64);
            this->RawSub(idxv, idxv, g, NR64); // idxv = idx mod N (robust: 0.5 bias margin >> fp err)
            this->RawCompareScalar(Ub<uint8_t>(pat0Addr_), idxv, 0.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat1Addr_), idxv, 1.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat2Addr_), idxv, 2.0f, CMPMODE::EQ, NR64);
            if constexpr (PACK5) {
                this->RawCompareScalar(Ub<uint8_t>(pat3Addr_), idxv, 3.0f, CMPMODE::EQ, NR64);
                this->RawCompareScalar(Ub<uint8_t>(pat4Addr_), idxv, 4.0f, CMPMODE::EQ, NR64);
            }
        }
        if constexpr (PACKN) {
            uint32_t Ncol = this->axis_;
            uint32_t NR = Ncol * R_;
            uint32_t NR64 = this->RoundUp(NR, 64u);
            uint32_t pb = this->RoundUp(NR, 256) / 8 + 32;
            col4Addr_ = Reserve(ub, rnc * sizeof(float));
            const4Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const4Addr_), 4.0f, rnc);
            col5Addr_ = Reserve(ub, rnc * sizeof(float));
            const5Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const5Addr_), 5.0f, rnc);
            col6Addr_ = Reserve(ub, rnc * sizeof(float));
            const6Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const6Addr_), 6.0f, rnc);
            col7Addr_ = Reserve(ub, rnc * sizeof(float));
            const7Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const7Addr_), 7.0f, rnc);
            col8Addr_ = Reserve(ub, rnc * sizeof(float));
            const8Addr_ = Reserve(ub, rnc * sizeof(float));
            this->RawDup(Ub<float>(const8Addr_), 8.0f, rnc);
            genAddr_ = Reserve(ub, NR64 * sizeof(float));
            gen2Addr_ = Reserve(ub, NR64 * sizeof(float));
            pat0Addr_ = Reserve(ub, pb);
            pat1Addr_ = Reserve(ub, pb);
            pat2Addr_ = Reserve(ub, pb);
            pat3Addr_ = Reserve(ub, pb);
            pat4Addr_ = Reserve(ub, pb);
            pat5Addr_ = Reserve(ub, pb);
            pat6Addr_ = Reserve(ub, pb);
            pat7Addr_ = Reserve(ub, pb);
            pat8Addr_ = Reserve(ub, pb);
            LocalTensor<float> idxv = Ub<float>(genAddr_);
            LocalTensor<float> g = Ub<float>(gen2Addr_);
            this->RawIota(idxv, 0.0f, NR64);
            this->RawAdds(g, idxv, 0.5f, NR64);
            this->RawMuls(g, g, 1.0f / (float)(int32_t)Ncol, NR64);
            this->template RawCast<RoundMode::CAST_FLOOR>(g.template ReinterpretCast<int32_t>(), g, NR64);
            this->template RawCast<RoundMode::CAST_NONE>(g, g.template ReinterpretCast<int32_t>(), NR64);
            this->RawMuls(g, g, (float)(int32_t)Ncol, NR64);
            this->RawSub(idxv, idxv, g, NR64); // idx mod N
            this->RawCompareScalar(Ub<uint8_t>(pat0Addr_), idxv, 0.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat1Addr_), idxv, 1.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat2Addr_), idxv, 2.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat3Addr_), idxv, 3.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat4Addr_), idxv, 4.0f, CMPMODE::EQ, NR64);
            this->RawCompareScalar(Ub<uint8_t>(pat5Addr_), idxv, 5.0f, CMPMODE::EQ, NR64);
            if (Ncol >= 7)
                this->RawCompareScalar(Ub<uint8_t>(pat6Addr_), idxv, 6.0f, CMPMODE::EQ, NR64);
            if (Ncol >= 8)
                this->RawCompareScalar(Ub<uint8_t>(pat7Addr_), idxv, 7.0f, CMPMODE::EQ, NR64);
            if (Ncol >= 9)
                this->RawCompareScalar(Ub<uint8_t>(pat8Addr_), idxv, 8.0f, CMPMODE::EQ, NR64);
        }
        if (!IsSmall()) {
            accValAddr_ = Reserve(ub, R8_ * sizeof(float));
            accIdxAddr_ = Reserve(ub, R8_ * sizeof(float));
            pieceValAddr_ = Reserve(ub, R8_ * sizeof(float));
            pieceIdxAddr_ = Reserve(ub, R8_ * sizeof(float));
        }
        if (!IsSmall() && !IsSeg()) {
            wchAddr_ = Reserve(ub, R8_ * sizeof(float)); // winning chunk, gather byte offsets, gathered local index
            offsAddr_ = Reserve(ub, R8_ * sizeof(int32_t));
            glocAddr_ = Reserve(ub, R8_ * sizeof(float));
            if (useHalf_) { // half reduces the value 128/chunk natively: per-chunk (value, int16-idx) scratch
                uint32_t rnc2 = this->RoundUp(R_ * nc_, 64);
                cminHAddr_ = Reserve(ub, rnc2 * sizeof(half));
                cidxHAddr_ = Reserve(ub, rnc2 * sizeof(half));
            }
        }
        if constexpr (SPLIT1) {
            workspaceValGm_ = reinterpret_cast<__gm__ float*>(ws);
            workspaceIdxGm_ = reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(ws) +
                                                                (uint64_t)usedCoreNum_ * SLOT_BYTES);
        } else if constexpr (SPLIT2) {
            valStride2d_ = this->RoundUp(this->outSize_, 8u);
            workspaceValGm_ = reinterpret_cast<__gm__ float*>(ws);
            workspaceArgGm_ = reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(ws) +
                                                                (uint64_t)usedCoreNum_ * valStride2d_ * sizeof(float));
        }
    }

    // ============================ Process: axis-path dispatch ============================
    // splitAxis_ (1 or 2) sends single-/multi-output huge-axis cases straight to the cross-core-combine
    // paths (see the "cross-core combine" section below); everything else picks exactly one of the 7
    // per-row axis-width paths (tiny / deint678 / deint5 / deint3 / deint / micro / seg), all of which share
    // the identical prefetch-compute-drain software pipeline via RunPipelined<>() -- only the Load/Compute
    // pair differs. The final >256 piece path is the odd one out: ComputeReduce prefetches its own pieces
    // internally, so it needs no outer Load/Compute split.
    __aicore__ inline void Process()
    {
        if constexpr (SPLIT1) {
            ProcessSplit();
        } else if constexpr (SPLIT2) {
            ProcessSplitMulti();
        } else {
            if constexpr (TINY)
                RunPipelined<&ArgLast::LoadTile, &ArgLast::ComputeTiny>();
            else if constexpr (PACK2)
                RunPipelined<&ArgLast::LoadMicro, &ArgLast::ComputeMicro>();
            else if constexpr (PACK3)
                RunPipelined<&ArgLast::LoadDeint, &ArgLast::ComputeDeinterleave3>();
            else if constexpr (PACK4)
                RunPipelined<&ArgLast::LoadDeint, &ArgLast::ComputeDeinterleave>();
            else if constexpr (PACK5)
                RunPipelined<&ArgLast::LoadDeint, &ArgLast::ComputeDeinterleave5>();
            else if constexpr (PACKN)
                RunPipelined<&ArgLast::LoadDeint, &ArgLast::ComputeDeinterleaveN678>();
            else if constexpr (SEG)
                RunPipelined<&ArgLast::LoadSeg, &ArgLast::ComputeSeg>();
            else
                RunPiece();
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

    __aicore__ inline LocalTensor<T> OutValue(uint32_t slot)
    {
        return LocalTensor<T>(TPosition::VECCALC, outValueAddr_ + slot * outValueSlotBytes_, outputCap_);
    }

    __aicore__ inline LocalTensor<int32_t> OutIndex(uint32_t slot)
    {
        return LocalTensor<int32_t>(TPosition::VECCALC, outIndexAddr_ + slot * outIndexSlotBytes_, outputCap_);
    }

    __aicore__ inline bool IsSmall() const
    {
        if constexpr (SPLIT2)
            return splitTiny_;
        return SMALL;
    }

    __aicore__ inline bool IsSeg() const
    {
        if constexpr (SPLIT2)
            return splitSeg_;
        return SEG;
    }

    __aicore__ inline bool IsPacked() const
    {
        if constexpr (SPLIT2)
            return splitMicro_;
        return PACK2 || PACK3 || PACK4 || PACK5 || PACKN;
    }

    __aicore__ inline void RunPiece()
    {
        for (uint32_t done = 0; done < this->oLen_; done += R_) {
            uint32_t R = (this->oLen_ - done) < R_ ? (this->oLen_ - done) : R_;
            if (done)
                this->template SyncMte3Complete<HardEvent::MTE3_V>();
            ComputeReduce(this->oStart_ + done, R, 0); // >256 piece: ComputeReduce prefetches its own pieces
            Drain(this->oStart_ + done, R, 0);
        }
    }
    // Shared software-pipeline loop for the 7 per-row axis paths dispatched above: prefetch tile N+1 under
    // tile N's compute, then drain, repeating across this core's output range. LoadFn/ComputeFn are fixed
    // template arguments (a specific path's member-function pair), so this resolves at compile time to the
    // exact same direct calls the 7 hand-written copies used to make -- purely a source-level dedup.
    template <void (ArgLast::*LoadFn)(uint32_t, uint32_t, uint32_t),
              void (ArgLast::*ComputeFn)(uint32_t, uint32_t, uint32_t)>
    __aicore__ inline void RunPipelined()
    {
        (this->*LoadFn)(this->oStart_, MinU(this->oLen_, R_), 0);
        uint32_t tile = 0;
        for (uint32_t done = 0; done < this->oLen_; done += R_, ++tile) {
            uint32_t R = (this->oLen_ - done) < R_ ? (this->oLen_ - done) : R_;
            uint32_t slot = tile & 1u;
            this->template Sync<HardEvent::MTE2_V>();
            uint32_t nd = done + R_;
            if (nd < this->oLen_) {
                if (tile > 0)
                    this->template Sync<HardEvent::V_MTE2>();
                (this->*LoadFn)(this->oStart_ + nd, (this->oLen_ - nd) < R_ ? (this->oLen_ - nd) : R_, slot ^ 1u);
            }
            if (tile > 1)
                this->template SyncMte3Complete<HardEvent::MTE3_V>();
            (this->*ComputeFn)(this->oStart_ + done, R, slot);
            Drain(this->oStart_ + done, R, slot);
        }
    }

    // ============================ shared small helpers ============================
    static __aicore__ inline uint32_t MinU(uint32_t a, uint32_t b) { return a < b ? a : b; }
    static __aicore__ inline uint32_t GcdU(uint32_t a, uint32_t b)
    {
        while (b != 0u) {
            uint32_t r = a % b;
            a = b;
            b = r;
        }
        return a;
    }
    __aicore__ inline uint32_t RoundUp512(uint32_t a) { return this->RoundUp(a, 512); }

    // ============================ TINY path (axis <= tinyLim) ============================
    // axis <= 64: load whole [R][axis] (row de-padded to W=RoundUp(axis,8)), then reduce values and indices
    // separately per <=252-row chunk and write directly to the outputs without a de-interleave step.
    __aicore__ inline void LoadTile(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        LocalTensor<T> x = Input(slot);
        if constexpr (TINY) {
            constexpr uint32_t BLK = 32u / sizeof(T);
            if (this->axis_ < BLK) {
                // A sub-datablock row is cheaper as one padding descriptor.  The aligned alternative would split
                // it into BLK/gcd(axis,BLK) separate MTE2 commands merely to manufacture a 32B row stride.
                this->LoadRowsNoFill(x, (uint64_t)rowBase * gmStride_ + sliceBase_, R, this->axis_, W_, gmStride_);
                return;
            }
            // Per-row padded copies are inefficient when many adjacent rows each carry a narrow, non-32-byte
            // block. Partition rows by the minimum period that makes the GM row step 32-byte aligned, then use the
            // aligned MTE2 form to fetch full datablocks.  Each group writes directly to its final row slots in UB;
            // WholeReduce masks lanes >=axis, so the over-read tail is dummy and needs no padding instruction.
            const uint16_t burst = static_cast<uint16_t>(W_ / BLK);
            const uint16_t srcStride = static_cast<uint16_t>(loadGroups_ * gmStride_ / BLK - burst);
            const uint16_t dstStride = static_cast<uint16_t>(loadGroups_ * W_ / BLK - burst);
            __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(x.GetPhyAddr());
            for (uint32_t group = 0; group < loadGroups_ && group < R; ++group) {
                const uint16_t rows = static_cast<uint16_t>((R - 1u - group) / loadGroups_ + 1u);
                copy_gm_to_ubuf(dst + group * W_, this->xGm_ + (uint64_t)(rowBase + group) * gmStride_ + sliceBase_, 0,
                                rows, burst, srcStride, dstStride);
            }
        } else {
            this->LoadRowsNoFill(x, (uint64_t)rowBase * gmStride_ + sliceBase_, R, this->axis_, W_, gmStride_);
        }
    }

    // ============================ SEG path (65 < axis <= 512) ============================
    // SEG load: no fill is needed because ReduceSeg never reads rowW padding, removing a prefill and V_MTE2 wait.
    __aicore__ inline void LoadSeg(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        LocalTensor<T> x = Input(slot);
        this->LoadRowsNoFill(x, (uint64_t)rowBase * gmStride_ + sliceBase_, R, this->axis_, W_, gmStride_);
    }

    // seg compute: consume the prefetched tile, reduce its single piece (pStart=0), write value+index outputs.
    // Bit-identical math to ComputeReduce's single-piece (axis<=256) path; split out only for the tile prefetch.
    __aicore__ inline void ComputeSeg(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase;
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<float> accVal = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                  Ub<float>(accValAddr_);
        LocalTensor<float> accIdx = Ub<float>(accIdxAddr_);
        LocalTensor<T> x = Input(slot);
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * W_);
        ReduceSeg(srcF, R, this->axis_, W_, 0, accVal, accIdx);
        LocalTensor<int32_t> oidx = OutIndex(slot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, accVal, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, accIdx, R);
    }
    // Reduce values and indices into separate contiguous outputs. ORDER_ONLY_INDEX writes uint32-width indices
    // regardless of source dtype, so the half path can target the int32 output storage without a data-moving cast.
    __aicore__ inline void ComputeTiny(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase; // load is prefetched by LoadTile in Process
        LocalTensor<T> x = Input(slot);
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<T> oval = OutValue(slot);
            LocalTensor<int32_t> oidx = OutIndex(slot);
            LocalTensor<T> oidxH = oidx.template ReinterpretCast<T>(); // uint32-index view, half-typed for the API
            // Two separate reductions minimize work for small row batches. For larger batches, one interleaved
            // reduction plus de-interleave avoids duplicating the reduction pass. GatherMask and Cast destinations
            // require 32-byte alignment, so the interleaved branch uses a 16-half-aligned chunk size. The index is
            // widened through half because the intrinsic does not provide a direct int16-to-int32 cast.
            constexpr uint32_t HALF_STRAT_A_CHUNK = 240;
            LocalTensor<T> redH = Ub<T>(redAddr_);
            LocalTensor<T> idxHalf = Ub<T>(cidxAddr_);
            uint64_t rsvd = 0;
            uint64_t lowMask = this->axis_ >= 64u ? ~0ULL : ((1ULL << this->axis_) - 1ULL);
            uint64_t highMask = this->axis_ <= 64u ?
                                    0ULL :
                                    (this->axis_ == 128u ? ~0ULL : ((1ULL << (this->axis_ - 64u)) - 1ULL));
            for (uint32_t r0 = 0; r0 < R;) {
                // GatherMask/Cast leave the vector mask registers in count-mask form.  Reload the row mask
                // before every chunk because vcmin/vcmax consume the current normal mask directly.
                set_mask_norm();
                set_vector_mask(highMask, lowMask);
                uint32_t remain = R - r0;
                uint32_t rc = remain < CHUNK_ROWS ? remain : CHUNK_ROWS;
                if (rc >= RC_STRATEGY_A_THRESHOLD) {
                    rc = remain < HALF_STRAT_A_CHUNK ? remain : HALF_STRAT_A_CHUNK;
                    this->template RawWholeReduce<Order_t::VALUE_INDEX>(redH, x[r0 * W_], static_cast<uint8_t>(rc), 1,
                                                                        1, W_ / 16);
                    this->RawGatherMask(oval[r0], redH, (uint8_t)1, true, 2 * rc, {1, 1, 8, 8}, rsvd);
                    this->RawGatherMask(oidxH[2 * r0], redH, (uint8_t)2, true, 2 * rc, {1, 1, 8, 8}, rsvd);
                    this->template RawCast<RoundMode::CAST_NONE>(idxHalf,
                                                                 oidxH[2 * r0].template ReinterpretCast<int16_t>(), rc);
                    this->template RawCast<RoundMode::CAST_RINT>(oidx[r0], idxHalf, rc);
                } else {
                    this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(oval[r0], x[r0 * W_],
                                                                              static_cast<uint8_t>(rc), 1, 1, W_ / 16);
                    this->template RawWholeReduce<Order_t::ONLY_INDEX>(oidxH[2 * r0], x[r0 * W_],
                                                                       static_cast<uint8_t>(rc), 1, 1, W_ / 16);
                }
                r0 += rc;
            }
            return;
        }
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * W_);
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<int32_t> oidx = OutIndex(slot);
        // Non-float T (bf16/int16) still needs an fp32-staged value for StoreValF's round-cast down to T;
        // float T writes straight into oval via reinterpret (both arms are LocalTensor<float>, resolved at
        // compile time by IsSameType, so this is a zero-cost type selection, not a runtime branch).
        LocalTensor<float> ovalF = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                 Ub<float>(cminAddr_);
        LocalTensor<float> oidxF = oidx.template ReinterpretCast<float>();
        // Separate reductions avoid de-interleave overhead for small row batches. Larger batches use one interleaved
        // reduction so the reduction pass is not duplicated; redF holds its value/index pairs before de-interleave.
        LocalTensor<float> redF = Ub<float>(redAddr_);
        // GatherMask and Adds require 32-byte-aligned LocalTensor operands, while WholeReduce permits a smaller
        // destination alignment. Keep the interleaved branch's chunk size a multiple of eight floats so every
        // de-interleaved output offset is 32-byte aligned. Other reduction paths retain CHUNK_ROWS.
        constexpr uint32_t TINY_STRAT_A_CHUNK = 248;
        uint64_t rsvd = 0;
        for (uint32_t r0 = 0; r0 < R;) {
            // RawGatherMask switches back to normal mode without restoring the mask bits.
            set_mask_norm();
            set_vector_mask(0, this->axis_ == 64u ? ~0ULL : ((1ULL << this->axis_) - 1ULL));
            uint32_t remain = R - r0;
            uint32_t rc = remain < CHUNK_ROWS ? remain : CHUNK_ROWS;
            if (rc >= RC_STRATEGY_A_THRESHOLD) {
                rc = remain < TINY_STRAT_A_CHUNK ? remain : TINY_STRAT_A_CHUNK;
                this->template RawWholeReduce<Order_t::VALUE_INDEX>(redF, srcF[r0 * W_], static_cast<uint8_t>(rc), 1, 1,
                                                                    W_ / 8);
                this->RawGatherMask(ovalF[r0], redF, (uint8_t)1, true, 2 * rc, {1, 1, 8, 8}, rsvd);
                this->RawGatherMask(oidxF[r0], redF, (uint8_t)2, true, 2 * rc, {1, 1, 8, 8}, rsvd);
            } else {
                this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(ovalF[r0], srcF[r0 * W_],
                                                                          static_cast<uint8_t>(rc), 1, 1, W_ / 8);
                this->template RawWholeReduce<Order_t::ONLY_INDEX>(oidxF[r0], srcF[r0 * W_], static_cast<uint8_t>(rc),
                                                                   1, 1, W_ / 8);
            }
            r0 += rc;
        }
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, ovalF, R);
    }

    // ============================ MICRO path (axis == 2) ============================
    // axis == 2: the [R][2] run is fully contiguous in GM, so load it in ONE block (MTE2 near peak instead of R
    // tiny per-row loads), split the two columns with this->RawGatherMask(patterns 1/2 = even/odd,
    // datablock-stride-free), then elementwise extremum selection with a first-occurrence index.
    __aicore__ inline void LoadMicro(uint32_t rowBase, uint32_t R, uint32_t slot)
    { // axis==2: the [R][2] run is contiguous in GM -> ONE packed block (MTE2 near peak)
        LocalTensor<T> x = Input(slot);
        __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(x.GetPhyAddr());
        __gm__ T* src = this->xGm_ + (uint64_t)rowBase * 2;
        if constexpr (sizeof(T) == 2u)
            copy_gm_to_ubuf_align_b16(dst, src, 0, 1, (uint64_t)R * 2 * sizeof(T), 0, 0, 0, 0);
        else
            copy_gm_to_ubuf_align_b32(dst, src, 0, 1, (uint64_t)R * 2 * sizeof(T), 0, 0, 0, 0);
    }
    // ============================ DE-INTERLEAVE paths (axis == 3, 4, 5, 6/7/8/9) ============================
    __aicore__ inline void LoadDeint(uint32_t rowBase, uint32_t R, uint32_t slot)
    { // axis (3-8): the [R][axis] run is contiguous in GM -> ONE packed block (MTE2 near peak, no per-row pad)
        LocalTensor<T> x = Input(slot);
        __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(x.GetPhyAddr());
        __gm__ T* src = this->xGm_ + (uint64_t)rowBase * this->axis_ + sliceBase_;
        if constexpr (sizeof(T) == 2u)
            copy_gm_to_ubuf_align_b16(dst, src, 0, 1, (uint64_t)R * this->axis_ * sizeof(T), 0, 0, 0, 0);
        else
            copy_gm_to_ubuf_align_b32(dst, src, 0, 1, (uint64_t)R * this->axis_ * sizeof(T), 0, 0, 0, 0);
    }
    // axis==3: packed contiguous load plus bitmask GatherMask de-interleave, followed by a three-way tournament.
    // axis 6/7/8: N user-pattern GatherMask operations produce N columns for an N-way tournament.
    __aicore__ inline void ComputeDeinterleaveN678(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase;
        LocalTensor<T> x = Input(slot);
        uint32_t Ncol = this->axis_;
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * Ncol);
        uint32_t rnc = this->RoundUp(R, 64);
        LocalTensor<float> cols[9] = {Ub<float>(cminAddr_), Ub<float>(cidxAddr_), Ub<float>(col2Addr_),
                                      Ub<float>(col3Addr_), Ub<float>(col4Addr_), Ub<float>(col5Addr_),
                                      Ub<float>(col6Addr_), Ub<float>(col7Addr_), Ub<float>(col8Addr_)};
        LocalTensor<float> ks[9] = {Ub<float>(const0Addr_), Ub<float>(const1Addr_), Ub<float>(const2Addr_),
                                    Ub<float>(const3Addr_), Ub<float>(const4Addr_), Ub<float>(const5Addr_),
                                    Ub<float>(const6Addr_), Ub<float>(const7Addr_), Ub<float>(const8Addr_)};
        LocalTensor<uint8_t> pats[9] = {Ub<uint8_t>(pat0Addr_), Ub<uint8_t>(pat1Addr_), Ub<uint8_t>(pat2Addr_),
                                        Ub<uint8_t>(pat3Addr_), Ub<uint8_t>(pat4Addr_), Ub<uint8_t>(pat5Addr_),
                                        Ub<uint8_t>(pat6Addr_), Ub<uint8_t>(pat7Addr_), Ub<uint8_t>(pat8Addr_)};
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<float> outF = oval.template ReinterpretCast<float>();
        LocalTensor<float> mA = Ub<float>(tmpAAddr_);
        LocalTensor<float> mB = Ub<float>(tmpAAddr_)[rnc];
        if constexpr (IsSameType<T, float>::value) {
            if (Ncol & 1u)
                mB = outF;
            else
                mA = outF;
        }
        LocalTensor<float> iA = Ub<float>(tmpBAddr_);
        LocalTensor<float> iB = Ub<float>(tmpBAddr_)[rnc];
        uint64_t rsvd = 0;
        for (uint32_t k = 0; k < Ncol; ++k)
            this->RawGatherMask(cols[k], srcF, pats[k].template ReinterpretCast<uint32_t>(), true, Ncol * R,
                                {1, 1, 8, 0}, rsvd);
        // N-way first-occurrence tournament with ping-pong value and index buffers.
        LocalTensor<float> curMin = cols[0];
        LocalTensor<float> curIdx = ks[0];
        for (uint32_t k = 1; k < Ncol; ++k) {
            LocalTensor<float> nMin = (k & 1u) ? mA : mB;
            LocalTensor<float> nIdx = (k & 1u) ? iA : iB;
            if constexpr (IS_MIN)
                this->RawMin(nMin, curMin, cols[k], R);
            else
                this->RawMax(nMin, curMin, cols[k], R);
            this->template RawCompareSelect<CMPMODE::EQ>(nIdx, curMin.template ReinterpretCast<int32_t>(),
                                                         nMin.template ReinterpretCast<int32_t>(), curIdx, ks[k],
                                                         rnc); // equal -> keep earlier index
            curMin = nMin;
            curIdx = nIdx;
        }
        LocalTensor<int32_t> oidx = OutIndex(slot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, curMin, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, curIdx, R);
    }
    // axis==5 packed de-interleave: five user-pattern GatherMask operations followed by a five-way tournament.
    __aicore__ inline void ComputeDeinterleave5(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase;
        LocalTensor<T> x = Input(slot);
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * 5u);
        uint32_t rnc = this->RoundUp(R, 64);
        LocalTensor<float> c0 = Ub<float>(cminAddr_);
        LocalTensor<float> c1 = Ub<float>(cidxAddr_);
        LocalTensor<float> c2 = Ub<float>(col2Addr_);
        LocalTensor<float> c3 = Ub<float>(col3Addr_);
        LocalTensor<float> c4 = Ub<float>(col4Addr_);
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<float> vals = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                Ub<float>(redAddr_);
        LocalTensor<float> idxF = Ub<float>(redAddr_)[rnc];
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        LocalTensor<float> m01 = Ub<float>(tmpAAddr_);
        LocalTensor<float> m23 = Ub<float>(tmpAAddr_)[rnc];
        LocalTensor<float> i01 = Ub<float>(tmpBAddr_);
        LocalTensor<float> i23 = Ub<float>(tmpBAddr_)[rnc];
        LocalTensor<float> k0 = Ub<float>(const0Addr_);
        LocalTensor<float> k1 = Ub<float>(const1Addr_);
        LocalTensor<float> k2 = Ub<float>(const2Addr_);
        LocalTensor<float> k3 = Ub<float>(const3Addr_);
        LocalTensor<float> k4 = Ub<float>(const4Addr_);
        uint64_t rsvd = 0;
        this->RawGatherMask(c0, srcF, Ub<uint8_t>(pat0Addr_).template ReinterpretCast<uint32_t>(), true, 5u * R,
                            {1, 1, 8, 0}, rsvd);
        this->RawGatherMask(c1, srcF, Ub<uint8_t>(pat1Addr_).template ReinterpretCast<uint32_t>(), true, 5u * R,
                            {1, 1, 8, 0}, rsvd);
        this->RawGatherMask(c2, srcF, Ub<uint8_t>(pat2Addr_).template ReinterpretCast<uint32_t>(), true, 5u * R,
                            {1, 1, 8, 0}, rsvd);
        this->RawGatherMask(c3, srcF, Ub<uint8_t>(pat3Addr_).template ReinterpretCast<uint32_t>(), true, 5u * R,
                            {1, 1, 8, 0}, rsvd);
        this->RawGatherMask(c4, srcF, Ub<uint8_t>(pat4Addr_).template ReinterpretCast<uint32_t>(), true, 5u * R,
                            {1, 1, 8, 0}, rsvd);
        // First-occurrence tournament; reuse c0/c1 for level 2 after they are consumed.
        if constexpr (IS_MIN) {
            this->RawMin(m01, c0, c1, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i01, c0.template ReinterpretCast<int32_t>(),
                                                         m01.template ReinterpretCast<int32_t>(), k0, k1, rnc);
            this->RawMin(m23, c2, c3, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i23, c2.template ReinterpretCast<int32_t>(),
                                                         m23.template ReinterpretCast<int32_t>(), k2, k3, rnc);
            this->RawMin(c0, m01, m23, R); // Reuse c0 for the final pairwise extremum.
            this->template RawCompareSelect<CMPMODE::EQ>(c1, m01.template ReinterpretCast<int32_t>(),
                                                         c0.template ReinterpretCast<int32_t>(), i01, i23,
                                                         rnc); // c1 reused = arg of c0..c3
            this->RawMin(vals, c0, c4, R);
            this->template RawCompareSelect<CMPMODE::EQ>(idxF, c0.template ReinterpretCast<int32_t>(),
                                                         vals.template ReinterpretCast<int32_t>(), c1, k4, rnc);
        } else {
            this->RawMax(m01, c0, c1, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i01, c0.template ReinterpretCast<int32_t>(),
                                                         m01.template ReinterpretCast<int32_t>(), k0, k1, rnc);
            this->RawMax(m23, c2, c3, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i23, c2.template ReinterpretCast<int32_t>(),
                                                         m23.template ReinterpretCast<int32_t>(), k2, k3, rnc);
            this->RawMax(c0, m01, m23, R);
            this->template RawCompareSelect<CMPMODE::EQ>(c1, m01.template ReinterpretCast<int32_t>(),
                                                         c0.template ReinterpretCast<int32_t>(), i01, i23, rnc);
            this->RawMax(vals, c0, c4, R);
            this->template RawCompareSelect<CMPMODE::EQ>(idxF, c0.template ReinterpretCast<int32_t>(),
                                                         vals.template ReinterpretCast<int32_t>(), c1, k4, rnc);
        }
        LocalTensor<int32_t> oidx = OutIndex(slot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, vals, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, idxF, R);
    }
    __aicore__ inline void ComputeDeinterleave3(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase;
        LocalTensor<T> x = Input(slot);
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * 3u);
        uint32_t rnc = this->RoundUp(R, 64);
        LocalTensor<float> c0 = Ub<float>(cminAddr_);
        LocalTensor<float> c1 = Ub<float>(cidxAddr_);
        LocalTensor<float> c2 = Ub<float>(col2Addr_);
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<float> vals = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                Ub<float>(redAddr_);
        LocalTensor<float> idxF = Ub<float>(redAddr_)[rnc];
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        LocalTensor<float> m01 = Ub<float>(tmpAAddr_);
        LocalTensor<float> i01 = Ub<float>(tmpBAddr_);
        LocalTensor<float> k0 = Ub<float>(const0Addr_);
        LocalTensor<float> k1 = Ub<float>(const1Addr_);
        LocalTensor<float> k2 = Ub<float>(const2Addr_);
        uint64_t rsvd = 0;
        // de-interleave 3 columns via precomputed bit-packed masks (user-defined GatherMask pattern)
        this->RawGatherMask(c0, srcF, Ub<uint8_t>(pat0Addr_).template ReinterpretCast<uint32_t>(), true, 3u * R,
                            {1, 1, 8, 0}, rsvd);
        this->RawGatherMask(c1, srcF, Ub<uint8_t>(pat1Addr_).template ReinterpretCast<uint32_t>(), true, 3u * R,
                            {1, 1, 8, 0}, rsvd);
        this->RawGatherMask(c2, srcF, Ub<uint8_t>(pat2Addr_).template ReinterpretCast<uint32_t>(), true, 3u * R,
                            {1, 1, 8, 0}, rsvd);
        // Three-way first-occurrence tournament without scalar Select or in-place operations.
        if constexpr (IS_MIN) {
            this->RawMin(m01, c0, c1, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i01, c0.template ReinterpretCast<int32_t>(),
                                                         m01.template ReinterpretCast<int32_t>(), k0, k1,
                                                         rnc); // c0==min01 ? 0 : 1
            this->RawMin(vals, m01, c2, R);
            this->template RawCompareSelect<CMPMODE::EQ>(idxF, m01.template ReinterpretCast<int32_t>(),
                                                         vals.template ReinterpretCast<int32_t>(), i01, k2,
                                                         rnc); // m01==rowMin ? i01 : 2
        } else {
            this->RawMax(m01, c0, c1, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i01, c0.template ReinterpretCast<int32_t>(),
                                                         m01.template ReinterpretCast<int32_t>(), k0, k1, rnc);
            this->RawMax(vals, m01, c2, R);
            this->template RawCompareSelect<CMPMODE::EQ>(idxF, m01.template ReinterpretCast<int32_t>(),
                                                         vals.template ReinterpretCast<int32_t>(), i01, k2, rnc);
        }
        LocalTensor<int32_t> oidx = OutIndex(slot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, vals, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, idxF, R);
    }
    // axis==4: packed contiguous load plus GatherMask de-interleave, followed by a four-way first-occurrence
    // tournament. The packed load avoids per-row padding.
    __aicore__ inline void ComputeDeinterleave(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase;
        LocalTensor<T> x = Input(slot);
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * 4u);
        uint32_t rnc = this->RoundUp(R, 64);
        LocalTensor<float> c0 = Ub<float>(cminAddr_);
        LocalTensor<float> c1 = Ub<float>(cidxAddr_);
        LocalTensor<float> c2 = Ub<float>(col2Addr_);
        LocalTensor<float> c3 = Ub<float>(col3Addr_);
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<float> vals = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                Ub<float>(redAddr_);
        LocalTensor<float> idxF = Ub<float>(redAddr_)[rnc];
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        LocalTensor<float> tA = Ub<float>(tmpAAddr_);
        LocalTensor<float> tB = Ub<float>(tmpBAddr_);
        uint64_t rsvd = 0;
        // axis=4 de-interleave via TWO levels of PROVEN even/odd splits (pattern 1=even, 2=odd).
        // Built-in patterns 3-6 are NOT element-stride-4. Nested even/odd reconstructs clean columns:
        // L1: even of packed [e0,e1,e2,e3] run -> tA=[e0,e2 per row]; odd -> tB=[e1,e3 per row].
        this->RawGatherMask(tA, srcF, (uint8_t)1, true, 4u * R, {1, 1, 8, 8}, rsvd);
        this->RawGatherMask(tB, srcF, (uint8_t)2, true, 4u * R, {1, 1, 8, 8}, rsvd);
        // L2: split each again -> c0=e0, c2=e2, c1=e1, c3=e3 (per row).
        this->RawGatherMask(c0, tA, (uint8_t)1, true, 2u * R, {1, 1, 8, 8}, rsvd);
        this->RawGatherMask(c2, tA, (uint8_t)2, true, 2u * R, {1, 1, 8, 8}, rsvd);
        this->RawGatherMask(c1, tB, (uint8_t)1, true, 2u * R, {1, 1, 8, 8}, rsvd);
        this->RawGatherMask(c3, tB, (uint8_t)2, true, 2u * R, {1, 1, 8, 8}, rsvd);
        // Hierarchical first-occurrence selection using tensor-tensor Select. Reuse tA/tB after L2 de-interleave
        // for pairwise value and index intermediates.
        LocalTensor<float> m01 = tA;
        LocalTensor<float> m23 = tA[rnc];
        LocalTensor<float> i01 = tB;
        LocalTensor<float> i23 = tB[rnc];
        LocalTensor<float> k0 = Ub<float>(const0Addr_);
        LocalTensor<float> k1 = Ub<float>(const1Addr_);
        LocalTensor<float> k2 = Ub<float>(const2Addr_);
        LocalTensor<float> k3 = Ub<float>(const3Addr_);
        if constexpr (IS_MIN) {
            this->RawMin(m01, c0, c1, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i01, c0.template ReinterpretCast<int32_t>(),
                                                         m01.template ReinterpretCast<int32_t>(), k0, k1,
                                                         rnc); // c0==min01 ? 0 : 1
            this->RawMin(m23, c2, c3, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i23, c2.template ReinterpretCast<int32_t>(),
                                                         m23.template ReinterpretCast<int32_t>(), k2, k3,
                                                         rnc); // c2==min23 ? 2 : 3
            this->RawMin(vals, m01, m23, R);
            this->template RawCompareSelect<CMPMODE::EQ>(idxF, m01.template ReinterpretCast<int32_t>(),
                                                         vals.template ReinterpretCast<int32_t>(), i01, i23,
                                                         rnc); // left pair wins ties -> first-occurrence
        } else {
            this->RawMax(m01, c0, c1, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i01, c0.template ReinterpretCast<int32_t>(),
                                                         m01.template ReinterpretCast<int32_t>(), k0, k1, rnc);
            this->RawMax(m23, c2, c3, R);
            this->template RawCompareSelect<CMPMODE::EQ>(i23, c2.template ReinterpretCast<int32_t>(),
                                                         m23.template ReinterpretCast<int32_t>(), k2, k3, rnc);
            this->RawMax(vals, m01, m23, R);
            this->template RawCompareSelect<CMPMODE::EQ>(idxF, m01.template ReinterpretCast<int32_t>(),
                                                         vals.template ReinterpretCast<int32_t>(), i01, i23, rnc);
        }
        LocalTensor<int32_t> oidx = OutIndex(slot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, vals, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, idxF, R);
    }
    // MICRO path.
    __aicore__ inline void ComputeMicro(uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        (void)rowBase; // Input was prefetched by LoadMicro.
        LocalTensor<T> x = Input(slot);
        LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * 2);
        LocalTensor<float> col0 = Ub<float>(cminAddr_);
        LocalTensor<float> col1 = Ub<float>(cidxAddr_);
        LocalTensor<float> red = Ub<float>(redAddr_);
        uint32_t rnc = this->RoundUp(R, 64);
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<float> vals = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() : red;
        LocalTensor<float> idxF = red[rnc];
        LocalTensor<float> ones = Ub<float>(oneAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        uint64_t rsvd = 0;
        this->RawGatherMask(col0, srcF, (uint8_t)1, true, 2 * R, {1, 1, 8, 8}, rsvd); // axis position 0 (even elements)
        this->RawGatherMask(col1, srcF, (uint8_t)2, true, 2 * R, {1, 1, 8, 8}, rsvd); // axis position 1 (odd elements)
        if constexpr (IS_MIN)
            this->RawMin(vals, col0, col1, R);
        else
            this->RawMax(vals, col0, col1, R);
        // Compare the selected value with col0 bitwise so signed-zero pairs follow the hardware-selected operand
        // instead of collapsing into an IEEE equality tie. An int32 equality compare avoids arithmetic and casts.
        LocalTensor<float> zeros = Ub<float>(cidxAddr_); // col1 finished with
        this->RawDup(zeros, 0.0f, rnc);
        this->template RawCompareSelect<CMPMODE::EQ>(idxF, col0.template ReinterpretCast<int32_t>(),
                                                     vals.template ReinterpretCast<int32_t>(), zeros, ones,
                                                     rnc); // col0 -> index 0, else index 1
        LocalTensor<int32_t> oidx = OutIndex(slot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, vals, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, idxF, R);
    }

    // ============================ PIECE path (axis > 256) ============================
    // axis > 256: walk the axis in <=PIECE_AXIS pieces, reduce each (batched), fold pieces strict-first.
    __aicore__ inline void ComputeReduce(uint32_t rowBase, uint32_t R, uint32_t outSlot)
    {
        LocalTensor<T> oval = OutValue(outSlot);
        LocalTensor<float> accVal = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                  Ub<float>(accValAddr_);
        LocalTensor<float> accIdx = Ub<float>(accIdxAddr_);
        LocalTensor<float> pVal = Ub<float>(pieceValAddr_);
        LocalTensor<float> pIdx = Ub<float>(pieceIdxAddr_);
        uint64_t baseOff = (uint64_t)rowBase * gmStride_ + sliceBase_;
        { // SW pipeline: prefetch piece 0 (MTE2 overlaps loop entry)
            uint32_t pLen0 = this->axis_ < PIECE_AXIS ? this->axis_ : PIECE_AXIS;
            uint32_t Wp0 = (pLen0 <= 256u) ? this->RoundUp(pLen0, 64) : RoundUp512(pLen0);
            LocalTensor<T> x0 = Input(0);
            this->LoadRows(x0, baseOff, R, pLen0, Wp0, gmStride_);
        }
        bool first = true;
        uint32_t piece = 0;
        for (uint32_t pStart = 0; pStart < this->axis_; pStart += PIECE_AXIS) {
            uint32_t pLen = (this->axis_ - pStart) < PIECE_AXIS ? (this->axis_ - pStart) : PIECE_AXIS;
            bool pieceSeg = (pLen <= 256u); // small piece / tiny tail -> tight Wp + ReduceSeg (handles nc<8 cheaply)
            uint32_t Wp = pieceSeg ? this->RoundUp(pLen, 64) : RoundUp512(pLen);
            uint32_t ncp = Wp / 64;
            LocalTensor<T> x = Input(piece & 1u);
            this->template Sync<HardEvent::MTE2_V>();
            uint32_t pStartN = pStart + PIECE_AXIS; // prefetch next piece: MTE2 load overlaps THIS piece's vec reduce
            if (pStartN < this->axis_) {
                uint32_t pLenN = (this->axis_ - pStartN) < PIECE_AXIS ? (this->axis_ - pStartN) : PIECE_AXIS;
                uint32_t WpN = (pLenN <= 256u) ? this->RoundUp(pLenN, 64) : RoundUp512(pLenN);
                if (piece > 0)
                    this->template Sync<HardEvent::V_MTE2>();
                LocalTensor<T> xN = Input((piece + 1u) & 1u);
                this->LoadRows(xN, baseOff + pStartN, R, pLenN, WpN, gmStride_);
            }
            LocalTensor<float> mVal = first ? accVal : pVal;
            LocalTensor<float> mIdx = first ? accIdx : pIdx;
            if constexpr (IsSameType<T, half>::value) {
                if (pieceSeg) {
                    LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * Wp);
                    ReduceSeg(srcF, R, pLen, Wp, pStart, mVal, mIdx);
                } else if (useHalf_ && Wp % 1024u == 0u) {
                    // native half: reduce the value 128/chunk (NO whole-input fp32 cast). Wp%1024==0 keeps
                    // nc128=Wp/128 a multiple of 8 so each row's chunk-mins are 32B-block-aligned for the level-2.
                    ReducePieceHalf(x, R, Wp / 128, pStart, mVal, mIdx);
                } else { // odd tail piece (Wp 512/1536/...): fall back to the float path -- rare and small
                    LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * Wp);
                    ReducePiece(srcF, R, ncp, pStart, mVal, mIdx);
                }
            } else {
                LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), R * Wp);
                if (pieceSeg)
                    ReduceSeg(srcF, R, pLen, Wp, pStart, mVal, mIdx);
                else
                    ReducePiece(srcF, R, ncp, pStart, mVal, mIdx);
            }
            if (first)
                first = false;
            else
                Fold(accVal, accIdx, pVal, pIdx, R);
            ++piece;
        }
        LocalTensor<int32_t> oidx = OutIndex(outSlot);
        if constexpr (!IsSameType<T, float>::value)
            this->StoreValF(oval, accVal, R);
        this->template RawCast<RoundMode::CAST_RINT>(oidx, accIdx, R);
    }

    // Rare ultra-long split fallback.  Piece-local indices remain fp32, but each one is converted immediately and
    // accumulated as int32, so a slice wider than 2^24 cannot round an odd winning position.  Ordinary split slices
    // stay on ComputeReduce/ProcessSplit and do not execute these extra conversions or folds.
    template <bool STORE_VALUE>
    __aicore__ inline LocalTensor<float> ComputeReduceExact(uint32_t rowBase, uint32_t R, uint32_t outSlot,
                                                            uint32_t loadSliceBase, uint32_t reduceAxis)
    {
        LocalTensor<T> oval = OutValue(outSlot);
        LocalTensor<float> accVal = IsSameType<T, float>::value ? oval.template ReinterpretCast<float>() :
                                                                  Ub<float>(accValAddr_);
        LocalTensor<int32_t> accIdx = OutIndex(outSlot);
        LocalTensor<float> pVal = Ub<float>(pieceValAddr_);
        LocalTensor<float> pIdx = Ub<float>(pieceIdxAddr_);
        LocalTensor<int32_t> candIdx = Ub<int32_t>(cidxAddr_);
        uint64_t baseOff = (uint64_t)rowBase * gmStride_ + loadSliceBase;
        uint32_t pLen0 = reduceAxis < PIECE_AXIS ? reduceAxis : PIECE_AXIS;
        this->LoadRows(Input(0), baseOff, R, pLen0, pLen0 <= 256u ? this->RoundUp(pLen0, 64u) : RoundUp512(pLen0),
                       gmStride_);
        bool first = true;
        uint32_t piece = 0;
        for (uint32_t pStart = 0; pStart < reduceAxis; pStart += PIECE_AXIS) {
            uint32_t pLen = (reduceAxis - pStart) < PIECE_AXIS ? (reduceAxis - pStart) : PIECE_AXIS;
            bool pieceSeg = pLen <= 256u;
            uint32_t Wp = pieceSeg ? this->RoundUp(pLen, 64u) : RoundUp512(pLen);
            LocalTensor<T> x = Input(piece & 1u);
            this->template Sync<HardEvent::MTE2_V>();
            uint32_t pStartN = pStart + PIECE_AXIS;
            if (pStartN < reduceAxis) {
                uint32_t pLenN = (reduceAxis - pStartN) < PIECE_AXIS ? (reduceAxis - pStartN) : PIECE_AXIS;
                if (piece > 0u)
                    this->template Sync<HardEvent::V_MTE2>();
                this->LoadRows(Input((piece + 1u) & 1u), baseOff + pStartN, R, pLenN,
                               pLenN <= 256u ? this->RoundUp(pLenN, 64u) : RoundUp512(pLenN), gmStride_);
            }
            LocalTensor<float> mVal = first ? accVal : pVal;
            LocalTensor<float> srcF;
            if constexpr (IsSameType<T, half>::value) {
                if (pieceSeg) {
                    srcF = this->ToF(x, Ub<float>(srcAddr_), R * Wp);
                    ReduceSeg(srcF, R, pLen, Wp, 0u, mVal, pIdx);
                } else if (useHalf_ && Wp % 1024u == 0u) {
                    ReducePieceHalf(x, R, Wp / 128u, 0u, mVal, pIdx);
                } else {
                    srcF = this->ToF(x, Ub<float>(srcAddr_), R * Wp);
                    ReducePiece(srcF, R, Wp / 64u, 0u, mVal, pIdx);
                }
            } else {
                srcF = this->ToF(x, Ub<float>(srcAddr_), R * Wp);
                if (pieceSeg)
                    ReduceSeg(srcF, R, pLen, Wp, 0u, mVal, pIdx);
                else
                    ReducePiece(srcF, R, Wp / 64u, 0u, mVal, pIdx);
            }
            LocalTensor<int32_t> nextIdx = first ? accIdx : candIdx;
            this->template RawCast<RoundMode::CAST_RINT>(nextIdx, pIdx, R);
            if (pStart != 0u)
                this->RawAdds(nextIdx, nextIdx, static_cast<int32_t>(pStart), R);
            if (first) {
                first = false;
            } else {
                this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(
                    accIdx.template ReinterpretCast<float>(), pVal, accVal, candIdx.template ReinterpretCast<float>(),
                    accIdx.template ReinterpretCast<float>(), this->RoundUp(R, 64u));
                if constexpr (IS_MIN)
                    this->RawMin(accVal, accVal, pVal, R);
                else
                    this->RawMax(accVal, accVal, pVal, R);
            }
            ++piece;
        }
        if constexpr (STORE_VALUE && !IsSameType<T, float>::value)
            this->StoreValF(oval, accVal, R);
        return accVal;
    }

    __aicore__ inline void ReduceSegChunk(const LocalTensor<float>& srcF, uint32_t R, uint32_t pLen, uint32_t Wp,
                                          uint32_t c, bool useSegB, const LocalTensor<float>& redc,
                                          const LocalTensor<float>& value, const LocalTensor<float>& index,
                                          uint64_t& rsvd)
    {
        uint32_t cl = pLen - c * 64;
        if (cl > 64)
            cl = 64;
        set_mask_norm();
        set_vector_mask(0, cl == 64u ? ~0ULL : ((1ULL << cl) - 1ULL));
        if (useSegB) {
            this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(value, srcF[c * 64], static_cast<uint8_t>(R), 1,
                                                                      1, Wp / 8);
            this->template RawWholeReduce<Order_t::ONLY_INDEX>(index, srcF[c * 64], static_cast<uint8_t>(R), 1, 1,
                                                               Wp / 8);
        } else {
            this->template RawWholeReduce<Order_t::VALUE_INDEX>(redc, srcF[c * 64], static_cast<uint8_t>(R), 1, 1,
                                                                Wp / 8);
            this->RawGatherMask(value, redc, (uint8_t)1, true, 2 * R, {1, 1, 8, 8}, rsvd);
            this->RawGatherMask(index, redc, (uint8_t)2, true, 2 * R, {1, 1, 8, 8}, rsvd);
        }
        this->template RawCast<RoundMode::CAST_NONE>(index, index.template ReinterpretCast<int32_t>(), R);
    }

    // 65..512: reduce per 64-chunk (narrow row), fold chunks with a strict compare (first chunk wins ties).
    __aicore__ inline void ReduceSeg(const LocalTensor<float>& srcF, uint32_t R, uint32_t pLen, uint32_t Wp,
                                     uint32_t pStart, const LocalTensor<float>& minF, const LocalTensor<float>& argF)
    {
        LocalTensor<float> redc = Ub<float>(redAddr_);
        LocalTensor<float> cmin = Ub<float>(cminAddr_);
        LocalTensor<float> cidx = Ub<float>(cidxAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        uint32_t cmpR = this->RoundUp(R, 64);
        uint64_t rsvd = 0;
        uint32_t nc = (Wp + 63) / 64;
        bool useSegB = nc >= 4u;
        ReduceSegChunk(srcF, R, pLen, Wp, 0, useSegB, redc, minF, argF, rsvd);
        for (uint32_t c = 1; c < nc; ++c) {
            ReduceSegChunk(srcF, R, pLen, Wp, c, useSegB, redc, cmin, cidx, rsvd);
            this->RawAdds(cidx, cidx, (float)(int32_t)(c * 64), R);
            this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(argF, cmin, minF, cidx, argF, cmpR);
            if constexpr (IS_MIN)
                this->RawMin(minF, minF, cmin, R);
            else
                this->RawMax(minF, minF, cmin, R);
        }
        if (pStart)
            this->RawAdds(argF, argF, (float)(int32_t)pStart, R); // tail piece: piece-local index -> global axis index
    }

    // > 256: ONE batched WholeReduce over all R*nc 64-chunks, then strided level-2 (value) + Broadcast/Compare/Select
    // (the earliest chunk attaining the extremum wins). minF/argF are piece-local; argF gets pStart added.
    __aicore__ inline void ReducePiece(const LocalTensor<float>& srcF, uint32_t R, uint32_t nc, uint32_t pStart,
                                       const LocalTensor<float>& minF, const LocalTensor<float>& argF)
    {
        LocalTensor<float> redc = Ub<float>(redAddr_);
        LocalTensor<float> cmin = Ub<float>(cminAddr_);
        LocalTensor<float> cidx = Ub<float>(cidxAddr_);
        uint64_t rsvd = 0;
        set_mask_norm();
        set_vector_mask(0, ~0ULL);
        for (uint32_t off = 0; off < R * nc; off += 254) {
            uint32_t rep = MinU(254u, R * nc - off);
            this->template RawWholeReduce<Order_t::VALUE_INDEX>(redc[2 * off], srcF[off * 64],
                                                                static_cast<uint8_t>(rep), 1, 1, 8);
        }
        this->RawGatherMask(cmin, redc, (uint8_t)1, true, 2 * R * nc, {1, 1, 8, 8}, rsvd);
        this->RawGatherMask(cidx, redc, (uint8_t)2, true, 2 * R * nc, {1, 1, 8, 8}, rsvd);
        this->template RawCast<RoundMode::CAST_NONE>(cidx, cidx.template ReinterpretCast<int32_t>(), R * nc);
        // cidx holds the per-chunk LOCAL index (0..63); PieceArgmin finds the winning chunk + gathers its local index.
        PieceArgmin(cmin, cidx, R, nc, 64u, pStart, minF, argF);
    }

    // Find each row's (rowMin, first-occurrence global index) from per-chunk (chunkMin, localIdx 0..chunkSize-1):
    // level-2 WholeReduce ORDER_VALUE_INDEX gives the FIRST chunk achieving rowMin (winChunk); Gather pulls that
    // chunk's local index; global = winChunk*chunkSize + localIdx[winChunk]. Replaces Broadcast/Compare/Select with
    // a short chain over R rather than R*nc.
    __aicore__ inline void PieceArgmin(const LocalTensor<float>& cmin, const LocalTensor<float>& lidx, uint32_t R,
                                       uint32_t nc, uint32_t chunkSize, uint32_t pStart, const LocalTensor<float>& minF,
                                       const LocalTensor<float>& argF)
    {
        LocalTensor<float> wchF = Ub<float>(wchAddr_);
        LocalTensor<int32_t> offs = Ub<int32_t>(offsAddr_);
        LocalTensor<float> gloc = Ub<float>(glocAddr_);
        set_mask_norm();
        set_vector_mask(0, nc == 64u ? ~0ULL : ((1ULL << nc) - 1ULL));
        this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(minF, cmin, static_cast<uint8_t>(R), 1, 1, nc / 8);
        this->template RawWholeReduce<Order_t::ONLY_INDEX>(wchF, cmin, static_cast<uint8_t>(R), 1, 1, nc / 8);
        // gather byte offset of the winning chunk's local index: (r*nc + winChunk[r]) * sizeof(float)
        this->RawIota(offs, (int32_t)0, R);
        this->RawMuls(offs, offs, (int32_t)nc, R);
        this->RawAdd(offs, offs, wchF.template ReinterpretCast<int32_t>(), R);
        this->RawMuls(offs, offs, (int32_t)sizeof(float), R);
        this->RawGather(gloc, lidx, offs.template ReinterpretCast<uint32_t>(), (uint32_t)0, R);
        // argF = winChunk*chunkSize + localIdx[winChunk] (+ pStart). reuse offs as the float winChunk after the gather.
        LocalTensor<float> wchFloat = Ub<float>(offsAddr_);
        this->template RawCast<RoundMode::CAST_NONE>(wchFloat, wchF.template ReinterpretCast<int32_t>(), R);
        this->RawMuls(wchFloat, wchFloat, (float)(int32_t)chunkSize, R); // aicore disallows uint->float; go via int32
        this->RawAdd(argF, wchFloat, gloc, R);
        if (pStart)
            this->RawAdds(argF, argF, (float)(int32_t)pStart, R);
    }

    // axis > 256, T == half: reduce the VALUE natively in half (128/chunk -- 2x denser than the float path, and NO
    // whole-input fp32 ToF cast over R*Wp elements). Only the small per-chunk results (R*nc) cast to float, then run
    // the proven index machinery (Select/WholeReduce need float for the < 2^24-exact arg). minF/argF out stay float,
    // so ComputeReduce/Fold/StoreValF are unchanged. Caller passes nc = Wp/128 (a multiple of 8 -- see the Wp%1024
    // guard).
    __aicore__ inline void ReducePieceHalf(const LocalTensor<half>& x, uint32_t R, uint32_t nc, uint32_t pStart,
                                           const LocalTensor<float>& minF, const LocalTensor<float>& argF)
    {
        LocalTensor<half> redH = Ub<half>(redAddr_);
        LocalTensor<half> cminH = Ub<half>(cminHAddr_);
        LocalTensor<half> cidxH = Ub<half>(cidxHAddr_);
        LocalTensor<float> cmin = Ub<float>(cminAddr_);
        LocalTensor<float> cidx = Ub<float>(cidxAddr_);
        uint64_t rsvd = 0;
        // Level-1: native half reduce over 128-element chunks (the whole-axis reduce; padded lanes are PadVal-inert).
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
        for (uint32_t off = 0; off < R * nc; off += 255) {
            uint32_t rep = MinU(255u, R * nc - off);
            this->template RawWholeReduce<Order_t::VALUE_INDEX>(redH[2 * off], x[off * 128], static_cast<uint8_t>(rep),
                                                                1, 1, 8);
        }
        this->RawGatherMask(cminH, redH, (uint8_t)1, true, 2 * R * nc, {1, 1, 8, 8}, rsvd); // chunk extrema
        this->RawGatherMask(cidxH, redH, (uint8_t)2, true, 2 * R * nc, {1, 1, 8, 8},
                            rsvd); // chunk local index (int16 bits)
        // Both conversions cover the same count and use the same 16->32-bit strides. Keep one counter-mask
        // scope for both independent vconv instructions, then wait once before PieceArgmin consumes either result.
        set_mask_count();
        set_vector_mask(0, R * nc);
        vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cmin.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ half*>(cminH.GetPhyAddr()), 1, 1, 1, 8, 4);
        vconv_s162f32(reinterpret_cast<__ubuf__ float*>(cidx.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int16_t*>(cidxH.GetPhyAddr()), 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
        // cidx holds the per-chunk LOCAL index (0..127); PieceArgmin finds the winning chunk + gathers its local index.
        PieceArgmin(cmin, cidx, R, nc, 128u, pStart, minF, argF);
    }

    // fold one piece result into the running accumulator with a strict compare (earlier piece wins ties).
    __aicore__ inline void Fold(const LocalTensor<float>& accVal, const LocalTensor<float>& accIdx,
                                const LocalTensor<float>& pVal, const LocalTensor<float>& pIdx, uint32_t R)
    {
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        uint32_t cmpR = this->RoundUp(R, 64);
        this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(accIdx, pVal, accVal, pIdx, accIdx, cmpR);
        if constexpr (IS_MIN)
            this->RawMin(accVal, accVal, pVal, R);
        else
            this->RawMax(accVal, accVal, pVal, R);
    }

    // ============================ shared output drain (all axis paths above) ============================
    __aicore__ inline void Drain(uint32_t outOff, uint32_t R, uint32_t slot)
    {
        Drain(outOff, R, OutValue(slot), OutIndex(slot));
    }

    __aicore__ inline void Drain(uint32_t outOff, uint32_t R, const LocalTensor<T>& oval,
                                 const LocalTensor<int32_t>& oidx)
    {
        if constexpr (GATHER) {
            // Fine-grained ranges are disjoint. Non-aligned MTE3 commits exactly blockLen bytes.
            this->template Sync<HardEvent::V_MTE3>();
            this->StoreOut(outOff, R, oval, oidx);
        } else {
            this->template Sync<HardEvent::V_MTE3>();
            this->StoreOut(outOff, R, oval, oidx);
        }
    }

    // ===== 1D axis-split (splitAxis_==1): single output + huge axis -> engage all cores on the one row. =====
    // Single output, huge axis: each core reduces its axis slice to a value and global-index partial, publishes it to
    // its own 32B workspace slot, then (after SyncAll) core 0 folds all partials with a strict compare (cores are
    // ordered by axis slice, so an earlier core holds the smaller index -> first-occurrence on ties).
    __aicore__ inline void ProcessSplit()
    {
        uint32_t core = GetBlockIdx();
        uint32_t aStart = core * axisPerCore_;
        uint32_t aLen = (aStart >= this->axis_) ?
                            0 :
                            ((this->axis_ - aStart) < axisPerCore_ ? (this->axis_ - aStart) : axisPerCore_);
        LocalTensor<float> accVal = Ub<float>(accValAddr_);
        LocalTensor<float> accIdx = Ub<float>(accIdxAddr_);
        LocalTensor<int32_t> localIdx = Ub<int32_t>(cidxAddr_);
        if (aLen > FP32_EXACT_INDEX_LIMIT) {
            accVal = ComputeReduceExact<false>(0u, 1u, 0u, aStart, aLen);
            localIdx = OutIndex(0);
        } else if (aLen == 0) {
            this->RawDupInitF(accVal, 1); // worse-than-any -> never wins the combine
            this->RawDup(accIdx, 0.0f, 1);
            this->RawDup(localIdx, (int32_t)0, 1);
        } else {
            LocalTensor<float> pVal = Ub<float>(pieceValAddr_);
            LocalTensor<float> pIdx = Ub<float>(pieceIdxAddr_);
            bool first = true;
            { // Prefetch piece 0 so its MTE2 overlaps loop entry.
                uint32_t pLen0 = aLen < PIECE_AXIS ? aLen : PIECE_AXIS;
                LocalTensor<T> x0 = Input(0);
                this->LoadRows(x0, (uint64_t)aStart, 1, pLen0, RoundUp512(pLen0), this->axis_);
            }
            uint32_t piece = 0;
            for (uint32_t pStart = 0; pStart < aLen; pStart += PIECE_AXIS) {
                uint32_t pLen = (aLen - pStart) < PIECE_AXIS ? (aLen - pStart) : PIECE_AXIS;
                uint32_t Wp = RoundUp512(pLen);
                uint32_t ncp = Wp / 64;
                LocalTensor<T> x = Input(piece & 1u);
                this->template Sync<HardEvent::MTE2_V>();
                uint32_t pStartN = pStart + PIECE_AXIS; // prefetch next piece: its MTE2 overlaps this piece's reduce
                if (pStartN < aLen) {
                    uint32_t pLenN = (aLen - pStartN) < PIECE_AXIS ? (aLen - pStartN) : PIECE_AXIS;
                    if (piece > 0)
                        this->template Sync<HardEvent::V_MTE2>();
                    LocalTensor<T> xN = Input((piece + 1u) & 1u);
                    this->LoadRows(xN, (uint64_t)aStart + pStartN, 1, pLenN, RoundUp512(pLenN), this->axis_);
                }
                LocalTensor<float> srcF = this->ToF(x, Ub<float>(srcAddr_), Wp);
                LocalTensor<float> mVal = first ? accVal : pVal;
                LocalTensor<float> mIdx = first ? accIdx : pIdx;
                ReducePiece(srcF, 1, ncp, pStart, mVal, mIdx); // keep the per-core index fp32-exact
                if (first)
                    first = false;
                else
                    Fold(accVal, accIdx, pVal, pIdx, 1);
                ++piece;
            }
            this->template RawCast<RoundMode::CAST_RINT>(localIdx, accIdx, 1);
        }
        if (aStart)
            this->RawAdds(localIdx, localIdx, static_cast<int32_t>(aStart), 1);
        this->template Sync<HardEvent::V_MTE3>();
        this->RawStore(workspaceValGm_ + (uint64_t)core * SLOT_ELEMS, accVal, sizeof(float));
        this->RawStore(workspaceIdxGm_ + (uint64_t)core * SLOT_ELEMS, localIdx, sizeof(int32_t));
        this->template SyncMte3Complete<HardEvent::MTE3_MTE2>();
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
        if (core == 0)
            Combine();
    }

    // Core 0 folds the usedCoreNum partials into the final value and index, then stores the single output.
    __aicore__ inline void Combine()
    {
        LocalTensor<float> curV = Ub<float>(accValAddr_);
        LocalTensor<int32_t> curI = Ub<int32_t>(cidxAddr_);
        LocalTensor<float> candV = Ub<float>(pieceValAddr_);
        LocalTensor<int32_t> candI = Ub<int32_t>(pieceIdxAddr_);
        LocalTensor<uint8_t> mask = Ub<uint8_t>(maskAddr_);
        // Slot 0 is this core's result and remains live in accVal/accIdx. Keep the local copy because its MTE3 write
        // need not be visible to a same-core MTE2 read-back at this point; only other cores' published slots are read.
        for (uint32_t c = 1; c < usedCoreNum_; ++c) {
            this->RawLoad(candV, workspaceValGm_ + (uint64_t)c * SLOT_ELEMS, sizeof(float));
            this->RawLoad(candI, workspaceIdxGm_ + (uint64_t)c * SLOT_ELEMS, sizeof(int32_t));
            this->template Sync<HardEvent::MTE2_V>();
            this->template RawCompareSelect<IS_MIN ? CMPMODE::LT : CMPMODE::GT>(
                curI.template ReinterpretCast<float>(), candV, curV, candI.template ReinterpretCast<float>(),
                curI.template ReinterpretCast<float>(), 64);
            if constexpr (IS_MIN)
                this->RawMin(curV, curV, candV, 1);
            else
                this->RawMax(curV, curV, candV, 1);
            this->template Sync<HardEvent::V_MTE2>(); // these reads must finish before the next load overwrites them
        }
        LocalTensor<T> oval = OutValue(0);
        if constexpr (IsSameType<T, float>::value)
            oval = curV.template ReinterpretCast<T>();
        else
            this->StoreValF(oval, curV, 1);
        Drain(0, 1, oval, curI);
    }

    // ===== 2D axis-split: multi-output + huge axis + few rows. =====
    // Each core reduces its slice [sliceBase_, +axisPerCore_) of EVERY output row (reusing the tiny/seg/piece
    // machinery via the gmStride_/sliceBase_-parameterised LoadRows), publishes (extremum, global-index) per row to
    // workspace block, then a distributed strict-first combine folds the usedCoreNum slices (earlier slice = smaller
    // axis index -> wins ties -> first occurrence).
    __aicore__ inline void ProcessSplitMulti()
    {
        uint32_t core = GetBlockIdx();
        for (uint32_t done = 0; done < this->outSize_; done += R_) {
            uint32_t R = (this->outSize_ - done) < R_ ? (this->outSize_ - done) : R_;
            if (this->axis_ > FP32_EXACT_INDEX_LIMIT) {
                ComputeReduceExact<true>(done, R, 0u, sliceBase_, this->axis_);
            } else if (splitMicro_) {
                LoadMicro(done, R, 0);
                this->template Sync<HardEvent::MTE2_V>();
                ComputeMicro(done, R, 0);
            } else if (splitTiny_) {
                LoadTile(done, R, 0);
                this->template Sync<HardEvent::MTE2_V>();
                ComputeTiny(done, R, 0);
            } else
                ComputeReduce(done, R, 0);
            DrainToWs(core, done, R, 0);
        }
        // This core's partial writes (MTE3) must COMPLETE before any combine read (MTE2) -- DMA GM access has no
        // cache-coherence issue, only pipe ordering; SyncAll then guarantees every core's writes land before reads.
        this->template Sync<HardEvent::MTE3_MTE2>();
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
        CombineMulti(core);
    }

    // Publish this slice's value and exact int32 global index. Keeping the index integer avoids fp32's 2^24 limit.
    __aicore__ inline void DrainToWs(uint32_t core, uint32_t rowBase, uint32_t R, uint32_t slot)
    {
        LocalTensor<T> oval = OutValue(slot);
        LocalTensor<int32_t> oidx = OutIndex(slot);
        LocalTensor<float> vF = oval.template ReinterpretCast<float>();
        if constexpr (!IsSameType<T, float>::value) {
            vF = Ub<float>(cminAddr_);
            this->template RawCast<RoundMode::CAST_NONE>(vF, oval, R);
        }
        if (sliceBase_)
            this->RawAdds(oidx, oidx, static_cast<int32_t>(sliceBase_), R);
        this->template Sync<HardEvent::V_MTE3>();
        this->RawStore(workspaceValGm_ + (uint64_t)core * valStride2d_ + rowBase, vF, R * sizeof(float));
        this->RawStore(workspaceArgGm_ + (uint64_t)core * valStride2d_ + rowBase, oidx, R * sizeof(int32_t));
        this->template SyncMte3Complete<HardEvent::MTE3_V>();
    }

    // this core folds the output rows [cb, cb+cr) it owns: read all usedCoreNum slices (strided) and keep the
    // strict-first (earliest slice) winner per row. Value + arg fold entirely in float (combine is tiny vs the reduce).
    __aicore__ inline void CombineMulti(uint32_t core)
    {
        uint32_t cr0 = this->RoundUp((this->outSize_ + usedCoreNum_ - 1) / usedCoreNum_, 8u);
        uint32_t cb = core * cr0;
        if (cb >= this->outSize_)
            return;
        uint32_t cr = (this->outSize_ - cb) < cr0 ? (this->outSize_ - cb) : cr0;
        LocalTensor<float> accV = Ub<float>(accValAddr_);
        LocalTensor<int32_t> accI = Ub<int32_t>(accIdxAddr_);
        LocalTensor<float> candV = Ub<float>(pieceValAddr_);
        LocalTensor<int32_t> candI = Ub<int32_t>(pieceIdxAddr_);
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
        LocalTensor<T> oval = OutValue(0);
        if constexpr (IsSameType<T, float>::value)
            oval = accV.template ReinterpretCast<T>();
        else
            this->StoreValF(oval, accV, cr);
        Drain(cb, cr, oval, accI);
    }

    constexpr static uint32_t PIECE_AXIS = 4096; // max elements reduced per piece (multiple of 512)
    constexpr static uint32_t FP32_EXACT_INDEX_LIMIT = 1u << 24;
    constexpr static uint32_t SLOT_BYTES = 32; // per-core workspace slot (no two cores share a 32B GM block)
    constexpr static uint32_t SLOT_ELEMS = SLOT_BYTES / sizeof(float);
    constexpr static uint32_t CHUNK_ROWS = 252; // WholeReduce repeat cap (<=255, mult of 4)
    constexpr static uint32_t
        RC_STRATEGY_A_THRESHOLD = 128;          // rc>=this -> ORDER_VALUE_INDEX+this->RawGatherMask(see ComputeTiny)
    constexpr static float SENTINEL_IDX = 1e9f; // larger than any represented index; marks an inactive slice

    uint32_t inputAddr_ = 0, inputSlotBytes_ = 0, inputCap_ = 0;
    uint32_t outValueAddr_ = 0, outValueSlotBytes_ = 0, outIndexAddr_ = 0, outIndexSlotBytes_ = 0, outputCap_ = 0;
    uint32_t srcAddr_ = 0, redAddr_ = 0, cminAddr_ = 0, cidxAddr_ = 0, maskAddr_ = 0, oneAddr_ = 0;
    uint32_t accValAddr_ = 0, accIdxAddr_ = 0, pieceValAddr_ = 0, pieceIdxAddr_ = 0;
    uint32_t cminHAddr_ = 0, cidxHAddr_ = 0, wchAddr_ = 0, offsAddr_ = 0, glocAddr_ = 0;
    __gm__ float* workspaceValGm_;
    __gm__ int32_t* workspaceIdxGm_;
    __gm__ int32_t* workspaceArgGm_; // 2D-split: exact per-core per-row indices, after the float value block
    uint32_t R_ = 1, W_ = 64, nc_ = 1, R8_ = 64, loadGroups_ = 1;
    uint32_t axisPerCore_ = 0, usedCoreNum_ = 1;
    uint32_t gmStride_ = 0, sliceBase_ = 0, valStride2d_ = 0; // 2D-split: full-axis row stride, slice start, ws stride
    bool splitTiny_ = false, splitSeg_ = false, splitMicro_ = false;
    bool useHalf_ = false, noSrcBuf_ = false;
    uint32_t col2Addr_ = 0, col3Addr_ = 0, col4Addr_ = 0, col5Addr_ = 0, col6Addr_ = 0;
    uint32_t col7Addr_ = 0, col8Addr_ = 0, tmpAAddr_ = 0, tmpBAddr_ = 0;
    uint32_t const0Addr_ = 0, const1Addr_ = 0, const2Addr_ = 0, const3Addr_ = 0, const4Addr_ = 0;
    uint32_t const5Addr_ = 0, const6Addr_ = 0, const7Addr_ = 0, const8Addr_ = 0;
    uint32_t pat0Addr_ = 0, pat1Addr_ = 0, pat2Addr_ = 0, pat3Addr_ = 0, pat4Addr_ = 0;
    uint32_t pat5Addr_ = 0, pat6Addr_ = 0, pat7Addr_ = 0, pat8Addr_ = 0, genAddr_ = 0, gen2Addr_ = 0;
};
} // namespace ArgWithValueNs
#endif // ARG_MAX_WITH_VALUE_LAST_H
