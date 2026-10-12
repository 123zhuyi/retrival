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
 * \file clip_by_norm_no_div_sum_kernel.h
 * \brief ClipByNormNoDivSum Kernel — ClipByNormNoDivSumKernel<T, RANK>
 */
#pragma once
#include "kernel_operator.h"
#include "clip_by_norm_no_div_sum_tiling_data.h"
#include "clip_by_norm_no_div_sum_struct.h"

// ============================================================
// Kernel 侧辅助函数
// ============================================================

__aicore__ inline void GetCoreRange(
    int64_t core_id, int64_t tiles_main, int64_t cores_tail, int64_t& start, int64_t& end)
{
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

__aicore__ inline int64_t GetUBSplitRange(int64_t a_o_off, int64_t a_o, int64_t a_i, int64_t a_i_tail)
{
    return (a_o_off == a_o - 1) ? a_i_tail : a_i;
}

__aicore__ inline bool FlatToEffectiveCoord(
    int64_t flat, const int64_t* max_bro_shape, int64_t rank, int64_t split_axis, int64_t a_i, int64_t a_o,
    int64_t* eff_coord)
{
    for (int64_t d = 0; d < rank; d++)
        eff_coord[d] = 0;
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
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;
}

__aicore__ inline int64_t CalcOutputOffset(const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;
}

__aicore__ inline int64_t CalcOutputTransferCount(
    const int64_t* normal_shape, int64_t rank, int64_t split_axis, int64_t a_i_seg)
{
    int64_t split_elems = (normal_shape[split_axis] == 1) ? 1 : a_i_seg;
    int64_t inner_elems = 1;
    for (int64_t d = split_axis + 1; d < rank; d++)
        inner_elems *= normal_shape[d];
    return split_elems * inner_elems;
}

// ============================================================
// VF 全链函数: 6 步计算（Compare → Select → Sqrt → Compare → Select → Max）
// ============================================================
template <typename T>
__simd_vf__ inline void ClipByNormNoDivSumVF(
    __ubuf__ T* dstAddr, // 输出 y
    __ubuf__ T* xAddr,   // 输入 x
    __ubuf__ T* gtAddr,  // 输入 greater_zeros
    __ubuf__ T* selAddr, // 输入 select_ones
    __ubuf__ T* maxAddr, // 输入 maximum_ones
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> xReg, gtReg, selReg, maxReg, yReg, tmpReg;
    AscendC::Reg::MaskReg mask, cmpMask1, cmpMask2;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);

        // Load 4 路输入
        AscendC::Reg::LoadAlign(xReg, xAddr, aReg);
        AscendC::Reg::LoadAlign(gtReg, gtAddr, aReg);
        AscendC::Reg::LoadAlign(selReg, selAddr, aReg);
        AscendC::Reg::LoadAlign(maxReg, maxAddr, aReg);

        // S1: cond1 = x > greater_zeros
        AscendC::Reg::Compare<T, AscendC::CMPMODE::GT>(cmpMask1, xReg, gtReg, mask);

        // S2: inner_sel = Select(cond1, x, select_ones)
        AscendC::Reg::Select<T>(yReg, xReg, selReg, cmpMask1);

        // S3: sqrt_val = Sqrt(inner_sel)
        AscendC::Reg::Sqrt<T, AscendC::Reg::MaskMergeMode::ZEROING>(yReg, yReg, mask);

        // S4: cond2 = x <= greater_zeros
        AscendC::Reg::Compare<T, AscendC::CMPMODE::LE>(cmpMask2, xReg, gtReg, mask);

        // S5: outer_sel = Select(cond2, x, sqrt_val)
        AscendC::Reg::Select<T>(yReg, xReg, yReg, cmpMask2);

        // S6: y = Max(outer_sel, maximum_ones)
        AscendC::Reg::Max<T, AscendC::Reg::MaskMergeMode::ZEROING>(yReg, yReg, maxReg, mask);

        // Store 输出
        AscendC::Reg::StoreAlign(dstAddr, yReg, aReg, mask);
    }
}

// ============================================================
// Kernel 主类
// ============================================================
template <typename T, int64_t RANK>
class ClipByNormNoDivSumKernel {
    // NDDMA 维度数 (最大5), RANK>5 时外层走 Flat loop
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;

    AscendC::TPipe pipe_;                             // 流水线管理器
    const ClipByNormNoDivSumTilingData<RANK>* td_;    // TilingData 指针
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];   // GM 输入
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots]; // GM 输出
    // Broadcast 算子一律用 TBuf 管理 UB 内存，禁止 TQue/裸指针
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];  // UB buffer (P=5 槽位)
    AscendC::MultiCopyParams<T, ND> nddmaParams_[kMaxInputSlots]; // 每输入一组 NDDMA 参数
    int64_t nddmaOuterIters_[kMaxInputSlots];                     // NDDMA 外循环次数
    int64_t nddma_dims_;                                          // 实际 NDDMA 维数

public:
    // GM 绑定 + TBuf 分配 + NDDMA 参数预计算
    __aicore__ inline void Init(
        GM_ADDR inputs[kMaxInputSlots], GM_ADDR outputs[kMaxOutputSlots], const ClipByNormNoDivSumTilingData<RANK>* td)
    {
        td_ = td;
        for (int i = 0; i < kMaxInputSlots; i++)
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        for (int i = 0; i < kMaxOutputSlots; i++)
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);

        // TBuf 初始化 (P=5 槽位, per_buf_bytes)
        for (int i = 0; i < kPhysNodes; i++)
            pipe_.InitBuffer(buf_[i], td_->per_buf_bytes);

        // NDDMA 参数预计算
        const int64_t* dstShape = td_->max_bro_shape;
        int64_t k = td_->split.axis;
        nddma_dims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        for (int inp = 0; inp < kMaxInputSlots; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->input_strides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
                inner *= (d == k) ? td_->split.a_i : dstShape[d];
                nd++;
            }
            for (; nd < ND; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
            }
            // outer loop 计算
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = k; d < RANK - nddma_dims_; d++)
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.a_i : dstShape[d];
        }
    }

    __aicore__ inline void Process()
    {
        int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main, td_->multicore.cores_tail, start, end);

        constexpr int UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4; // P=5 槽位
        constexpr int IN_X = 0, IN_GT = 1, IN_SEL = 2, IN_MAX = 3; // 输入索引
        constexpr int OUT_Y = 0;                                   // 输出索引

        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg =
                GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o, td_->split.a_i, td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(
                flat, td_->max_bro_shape, RANK, td_->split.axis, td_->split.a_i, td_->split.a_o, coord);

            // 跨迭代 WAR: 上轮 CopyOut(MTE3) → 本轮 CopyIn(MTE2)
            if (flat != start)
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // === CopyIn: GM→UB (NDDMA 4 路) ===
            CopyInBrc(coord, IN_X, UB0, a_i_seg);
            CopyInBrc(coord, IN_GT, UB1, a_i_seg);
            CopyInBrc(coord, IN_SEL, UB2, a_i_seg);
            CopyInBrc(coord, IN_MAX, UB3, a_i_seg);

            // RAW: MTE2 写完 UB → V 要读
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

            // === VF 全链计算: Compare→Select→Sqrt→Compare→Select→Max ===
            ClipByNormNoDivSumVF<T>(
                (__ubuf__ T*)buf_[UB4].Get<T>().GetPhyAddr(), // dst: y
                (__ubuf__ T*)buf_[UB0].Get<T>().GetPhyAddr(), // src: x
                (__ubuf__ T*)buf_[UB1].Get<T>().GetPhyAddr(), // src: greater_zeros
                (__ubuf__ T*)buf_[UB2].Get<T>().GetPhyAddr(), // src: select_ones
                (__ubuf__ T*)buf_[UB3].Get<T>().GetPhyAddr(), // src: maximum_ones
                count, AscendC::GetVecLen() / sizeof(T),
                AscendC::CeilDivision(count, AscendC::GetVecLen() / sizeof(T)));

            // WAR: V 写完 UB4 → MTE3 可读
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            // === CopyOut: UB→GM (单输出) ===
            CopyOutOne(coord, OUT_Y, UB4, a_i_seg);

            if (flat != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    // CopyInBrc — NDDMA 随路 broadcast 搬运
    __aicore__ inline void CopyInBrc(const int64_t* coord, int inputIdx, int slot, int64_t a_i_seg)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcInputOffset(coord, td_->input_strides[inputIdx], RANK);
        const int64_t* dstShape = td_->max_bro_shape;

        auto params = nddmaParams_[inputIdx];
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd)
                params.loopInfo.loopSize[nd] = a_i_seg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {
            false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad, false};

        if constexpr (RANK <= 5) {
            AscendC::DataCopy<T, ND, cfg>(buf_[slot].Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].Get<T>();
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

    // CopyOut — DataCopyPad 非对齐回写
    __aicore__ inline void CopyOutOne(const int64_t* coord, int outputIdx, int slot, int64_t a_i_seg)
    {
        int64_t off = CalcOutputOffset(coord, td_->output_strides[outputIdx], RANK);
        int64_t cnt = CalcOutputTransferCount(td_->output_shapes[outputIdx], RANK, td_->split.axis, a_i_seg);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].Get<T>(), extParams);
    }
};
