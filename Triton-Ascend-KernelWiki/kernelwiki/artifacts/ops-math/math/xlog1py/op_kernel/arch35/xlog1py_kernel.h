/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// Xlog1py Kernel — 2入1出 broadcast, VF 全链融合 (Adds→Ln→Mul→Compare→Select→Compare→Select)
// 链长=7, P_FP32=3 P_FP16=4
#pragma once
#include "kernel_operator.h"
#include "xlog1py_tiling_struct.h"
#include "xlog1py_struct.h"

__aicore__ inline void GetCoreRange(int64_t core_id, int64_t tiles_main, int64_t cores_tail,
    int64_t& start, int64_t& end)
{
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

__aicore__ inline int64_t GetUBSplitRange(
    int64_t a_o_off, int64_t a_o, int64_t a_i, int64_t a_i_tail)
{
    return (a_o_off == a_o - 1) ? a_i_tail : a_i;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* max_bro_shape,
    int64_t rank, int64_t split_axis, int64_t a_i, int64_t a_o, int64_t* eff_coord)
{
    for (int64_t d = 0; d < rank; d++) eff_coord[d] = 0;
    int64_t a_o_off = flat % a_o;
    int64_t outer = flat / a_o;
    for (int64_t d = split_axis - 1; d >= 0; d--) {
        eff_coord[d] = outer % max_bro_shape[d];
        outer /= max_bro_shape[d];
    }
    eff_coord[split_axis] = a_o_off * a_i;
    return true;
}

__aicore__ inline int64_t CalcInputOffset(const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) offset += eff_coord[d] * strides[d];
    return offset;
}

__aicore__ inline int64_t CalcOutputOffset(const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) offset += eff_coord[d] * strides[d];
    return offset;
}

__aicore__ inline int64_t CalcInputTransferCount(
    const int64_t* normal_shape, int64_t rank, int64_t split_axis, int64_t a_i_seg)
{
    int64_t split_elems = (normal_shape[split_axis] == 1) ? 1 : a_i_seg;
    int64_t inner_elems = 1;
    for (int64_t d = split_axis + 1; d < rank; d++) inner_elems *= normal_shape[d];
    return split_elems * inner_elems;
}

__aicore__ inline int64_t CalcOutputTransferCount(
    const int64_t* normal_shape, int64_t rank, int64_t split_axis, int64_t a_i_seg)
{
    int64_t split_elems = (normal_shape[split_axis] == 1) ? 1 : a_i_seg;
    int64_t inner_elems = 1;
    for (int64_t d = split_axis + 1; d < rank; d++) inner_elems *= normal_shape[d];
    return split_elems * inner_elems;
}

template <typename T>
__simd_vf__ inline void Xlog1pyVF(
    __ubuf__ T* dstAddr, __ubuf__ T* srcXAddr, __ubuf__ T* srcYAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes,
    float scalar_one);

// ============================================================
// Xlog1pyKernel
// ============================================================

template <typename T, int64_t RANK>
class Xlog1pyKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr bool    NEED_CAST = !std::is_same_v<T, float>;
    static constexpr int64_t NUM_BUF = NEED_CAST ? 4 : 3;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);

    AscendC::TPipe pipe_;
    const Xlog1pyTilingData<RANK>* td_;
    AscendC::GlobalTensor<T>  gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T>  gmOut_[kMaxOutputSlots];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[NUM_BUF];
    AscendC::MultiCopyParams<T, ND> nddmaParams_[kMaxInputSlots];
    int64_t nddmaOuterIters_[kMaxInputSlots];
    int64_t nddma_dims_;

public:
    __aicore__ inline void Init(GM_ADDR* inputs, GM_ADDR* outputs,
                                const Xlog1pyTilingData<RANK>* td)
    {
        td_ = td;
        for (int i = 0; i < kMaxInputSlots; i++)
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        for (int i = 0; i < kMaxOutputSlots; i++)
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        for (int i = 0; i < NUM_BUF; i++)
            pipe_.InitBuffer(buf_[i], td_->per_buf_bytes);

        const int64_t* dstShape = td_->max_bro_shape;
        int64_t k = td_->split.axis;
        nddma_dims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        for (int inp = 0; inp < kMaxInputSlots; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd]      = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->input_strides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd]     = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd]     = 0;
                inner *= (d == k) ? td_->split.a_i : dstShape[d];
                nd++;
            }
            for (; nd < ND; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd]      = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd]     = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd]     = 0;
            }
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = k; d < RANK - nddma_dims_; d++)
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.a_i : dstShape[d];
        }
    }

    __aicore__ inline void Process()
    {
        int32_t evMTE2toV    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE2    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        int32_t evVtoMTE3    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main,
                     td_->multicore.cores_tail, start, end);

        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg = GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o,
                                              td_->split.a_i, td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(flat, td_->max_bro_shape, RANK,
                                 td_->split.axis, td_->split.a_i, td_->split.a_o, coord);

            if (flat != start) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            constexpr int IN_X = 0, IN_Y = 1;
            constexpr int OUT_Z = 0;

            if constexpr (NEED_CAST) {
                ProcessFP16(coord, IN_X, IN_Y, OUT_Z, a_i_seg, count,
                            evMTE2toV, evVtoMTE2, evVtoMTE3);
            } else {
                ProcessFP32(coord, IN_X, IN_Y, OUT_Z, a_i_seg, count,
                            evMTE2toV, evVtoMTE2, evVtoMTE3);
            }

            if (flat != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    __aicore__ inline void ProcessFP32(
        const int64_t* coord, int inX, int inY, int outZ,
        int64_t a_i_seg, int64_t count,
        int32_t evMTE2toV, int32_t evVtoMTE2, int32_t evVtoMTE3)
    {
        constexpr int B_X = 0, B_Y = 1, B_Z = 2;
        CopyInOne(coord, inX, B_X, a_i_seg);
        CopyInOne(coord, inY, B_Y, a_i_seg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        CallXlog1pyVF(B_Z, B_X, B_Y, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
        CopyOutOne(coord, outZ, B_Z, a_i_seg);
    }

    __aicore__ inline void ProcessFP16(
        const int64_t* coord, int inX, int inY, int outZ,
        int64_t a_i_seg, int64_t count,
        int32_t evMTE2toV, int32_t evVtoMTE2, int32_t evVtoMTE3)
    {
        constexpr int B_TEMP_X = 0, B_X = 1, B_Y = 2, B_TEMP_Y = 3;
        CopyInOne(coord, inX, B_TEMP_X, a_i_seg);
        CopyInOne(coord, inY, B_TEMP_Y, a_i_seg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        AscendC::Cast(buf_[B_X].template Get<float>(), buf_[B_TEMP_X].template Get<T>(),
                      AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(buf_[B_Y].template Get<float>(), buf_[B_TEMP_Y].template Get<T>(),
                      AscendC::RoundMode::CAST_NONE, count);
        CallXlog1pyVF(B_X, B_X, B_Y, count);
        AscendC::Cast(buf_[B_TEMP_X].template Get<T>(), buf_[B_X].template Get<float>(),
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
        CopyOutOne(coord, outZ, B_TEMP_X, a_i_seg);
    }

    __aicore__ inline void CallXlog1pyVF(int dstSlot, int srcXSlot, int srcYSlot, int64_t count)
    {
        uint16_t repeatTimes = AscendC::CeilDivision(count, VL);
        __ubuf__ float* dstAddr  = (__ubuf__ float*)buf_[dstSlot].template Get<float>().GetPhyAddr();
        __ubuf__ float* srcXAddr = (__ubuf__ float*)buf_[srcXSlot].template Get<float>().GetPhyAddr();
        __ubuf__ float* srcYAddr = (__ubuf__ float*)buf_[srcYSlot].template Get<float>().GetPhyAddr();
        asc_vf_call<Xlog1pyVF<float>>(dstAddr, srcXAddr, srcYAddr,
                                      (uint32_t)count, VL, repeatTimes, 1.0f);
    }

    __aicore__ inline void CopyInOne(const int64_t* coord, int inputIdx, int slot, int64_t a_i_seg)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcInputOffset(coord, td_->input_strides[inputIdx], RANK);
        const int64_t* dstShape = td_->max_bro_shape;
        auto params = nddmaParams_[inputIdx];
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd) params.loopInfo.loopSize[nd] = a_i_seg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        if constexpr (RANK <= 5) {
            AscendC::DataCopy<T, ND, cfg>(buf_[slot].template Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].template Get<T>();
            int64_t elem_base = off;
            for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
                int64_t elem_adj = 0, tmp = oi;
                for (int64_t d = RANK - nddma_dims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? a_i_seg : dstShape[d];
                    elem_adj += (tmp % sz) * td_->input_strides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, ND, cfg>(buf[oi * inner], gmIn_[inputIdx][elem_base + elem_adj], params);
            }
        }
    }

    __aicore__ inline void CopyOutOne(const int64_t* coord, int outputIdx, int slot, int64_t a_i_seg)
    {
        int64_t off = CalcOutputOffset(coord, td_->output_strides[outputIdx], RANK);
        int64_t cnt = CalcOutputTransferCount(td_->output_shapes[outputIdx], RANK,
                                              td_->split.axis, a_i_seg);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen   = cnt * sizeof(T);
        extParams.srcStride  = 0;
        extParams.dstStride  = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].template Get<T>(), extParams);
    }
};

// ============================================================
// Xlog1pyVF — Adds→Ln→Mul→Compare→Select→Compare→Select (链长=7)
// ============================================================

template <typename T>
__simd_vf__ inline void Xlog1pyVF(
    __ubuf__ T* dstAddr, __ubuf__ T* srcXAddr, __ubuf__ T* srcYAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes,
    float scalar_one)
{
    AscendC::Reg::RegTensor<T> regX, regY, regYp1, regLogY, regMul, regRes0, regFinal, regZero;
    AscendC::Reg::MaskReg mask, maskEq, maskNaN;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);

        AscendC::Reg::LoadAlign(regX, srcXAddr, aReg);
        AscendC::Reg::LoadAlign(regY, srcYAddr, aReg);

        // S1: Adds(y, 1.0) → y+1
        AscendC::Reg::Adds(regYp1, regY, scalar_one, mask);

        // S2: Ln(y+1) → log(1+y)
        AscendC::Reg::Ln(regLogY, regYp1, mask);

        // S3: Mul(x, log_y) → mul_res
        AscendC::Reg::Mul(regMul, regX, regLogY, mask);

        // S4: Compare(x, 0, EQ) → mask_eq
        AscendC::Reg::Sub(regZero, regX, regX, mask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(maskEq, regX, regZero, mask);

        // S5: Select(mask_eq, 0, mul_res) → res0
        AscendC::Reg::Select<float>(regRes0, regZero, regMul, maskEq);

        // S6: Compare(y, y, EQ) → mask_nan (NaN≠NaN → mask=0)
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(maskNaN, regY, regY, mask);

        // S7: Select(mask_nan, res0, y) → final
        AscendC::Reg::Select<float>(regFinal, regRes0, regY, maskNaN);

        AscendC::Reg::StoreAlign(dstAddr, regFinal, aReg, mask);
    }
}
