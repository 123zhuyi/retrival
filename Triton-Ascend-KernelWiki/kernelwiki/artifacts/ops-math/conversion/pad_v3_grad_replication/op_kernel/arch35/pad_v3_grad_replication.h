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
 * \file pad_v3_grad_replication.h
 * \brief pad_v3_grad_replication kernel (arch35)
 */

#ifndef PAD_V3_GRAD_REPLICATION_H
#define PAD_V3_GRAD_REPLICATION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pad_v3_grad_replication_struct.h"

namespace PadV3GradReplication {
using namespace AscendC;

constexpr uint32_t MAX_DIMS = PAD_GRAD_REPLICATION_MAX_DIMS_NUM;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VREG_BYTES = 256;                                             // 向量寄存器宽度（arch35）
constexpr uint8_t MAX_PAD_DIMS = (uint8_t)PAD_GRAD_REPLICATION_MAX_PAD_DIMS_NUM; // 后5维可做padding的最大维度数
constexpr uint32_t DIM_TWO = 2;
constexpr uint8_t NUM_TWO = 2;

// 精度提升类型：FP16/BF16 → FP32，其它类型保持原样
template <typename T>
struct PromoteOf {
    using type = T;
};
template <>
struct PromoteOf<half> {
    using type = float;
};
template <>
struct PromoteOf<bfloat16_t> {
    using type = float;
};

template <typename T>
__aicore__ inline constexpr bool NeedCast()
{
    return IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value;
}

// vec_scope cast traits
// 对宽度不同（B16 ↔ B32）类型，必须 ZERO + ONE 两次 cast + Select 合并，否则有数据丢失
// 参考 conversion/batch_to_space_nd/op_kernel/arch35/batch_to_space_nd_small_c.h
constexpr static Reg::CastTrait CAST_TRAIT_PROMOTE_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static Reg::CastTrait CAST_TRAIT_PROMOTE_ONE = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static Reg::CastTrait CAST_TRAIT_DOWN_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr static Reg::CastTrait CAST_TRAIT_DOWN_ONE = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    return (b == 0) ? a : ((a + b - 1) / b * b);
}

// =====================================================================
// Kernel 主体
// 模板参数：
//   T          - 输入/输出数据类型
//   DIM_NUM    - 维度数（1~8），编译期常量
//   SPLIT_AXIS - 切分轴（0~DIM_NUM-2），编译期常量；尾轴由 edge_simt 单独处理
// =====================================================================
template <typename T, uint8_t DIM_NUM, uint8_t SPLIT_AXIS>
class KernelPadV3GradReplication {
public:
    using PromoteT = typename PromoteOf<T>::type;
    static constexpr bool kNeedCast = NeedCast<T>();

    __aicore__ inline KernelPadV3GradReplication(TPipe* pipe, const PadV3GradReplicationTilingData* td)
        : pipe_(pipe), tilingData_(td)
    {}

    // ---------------- Init ----------------
    __aicore__ inline void Init(GM_ADDR gradOut, GM_ADDR gradIn)
    {
        blockIdx_ = GetBlockIdx();
        gradOutGm_.SetGlobalBuffer((__gm__ T*)gradOut);
        gradInGm_.SetGlobalBuffer((__gm__ T*)gradIn);

        for (uint32_t k = 0; k < DIM_NUM; k++) {
            inputShape_[k] = tilingData_->inputShape[k];
            outputShape_[k] = tilingData_->outputShape[k];
            strideAligned_[k] = tilingData_->strideAligned[k];
            leftPad_[k] = (int32_t)tilingData_->leftPad[k];
            rightPad_[k] = (int32_t)tilingData_->rightPad[k];
        }
        splitSize_ = tilingData_->splitSize;
        splitCount_ = tilingData_->splitCount;
        tilesPerCore_ = tilingData_->tilesPerCore;

        splitCountAxis_ = CeilDiv(inputShape_[SPLIT_AXIS], (uint64_t)splitSize_);

        // worstFactor：外层 corner tile 的 UB 放大因子
        worstFactor_ = 1;
        for (uint32_t k = 0; k < SPLIT_AXIS; k++) {
            if (inputShape_[k] == 1 && leftPad_[k] > 0 && rightPad_[k] > 0) {
                worstFactor_ *= ((uint64_t)leftPad_[k] + (uint64_t)rightPad_[k] + 1);
            } else {
                uint64_t maxPad = (leftPad_[k] > rightPad_[k] ? leftPad_[k] : rightPad_[k]);
                worstFactor_ *= (maxPad + 1);
            }
        }

        // 内层 axis > SPLIT_AXIS 部分的 element 数。
        //   innerProdUb_：UB 内（尾轴 32B 对齐 + 外层 extent-packed）—— 用于 dataBuf 大小估算、
        //                AccumNonLastAxis 与 SPLIT_AXIS 切片间距。SPLIT_AXIS<N-1 时 = strideAligned_[SPLIT_AXIS]。
        //   innerProdInGm_：grad_in 侧 GM 紧凑（inputShape 累乘）—— 用于写回 GM。
        innerProdUb_ = strideAligned_[SPLIT_AXIS];
        innerProdInGm_ = 1;
        for (uint32_t k = SPLIT_AXIS + 1; k < DIM_NUM; k++) {
            innerProdInGm_ *= inputShape_[k];
        }
        // dataBuf 容量：worstFactor × (splitSize + padOverhead) × innerProdUb
        // padOverhead：单 tile 需两端 pad(pL+pR)，多 tile 每 tile 至多一端(max(pL,pR))
        // dataBuf 布局：尾轴 32B 对齐（strideAligned_[N-2]），外层轴 0..SPLIT_AXIS-1 按 extent 紧密排布。
        bool isSingleTile = ((uint64_t)splitSize_ == inputShape_[SPLIT_AXIS]);
        uint64_t padOverhead = isSingleTile ? (leftPad_[SPLIT_AXIS] + rightPad_[SPLIT_AXIS]) :
                                              (leftPad_[SPLIT_AXIS] > rightPad_[SPLIT_AXIS] ? leftPad_[SPLIT_AXIS] :
                                                                                              rightPad_[SPLIT_AXIS]);
        uint64_t maxSliceAxis = (uint64_t)splitSize_ + padOverhead;
        uint64_t dataBufElems = worstFactor_ * maxSliceAxis * innerProdUb_;
        dataBufElems_ = dataBufElems; // BulkCast 用

        // outputBuf 容量：splitSize × innerProdInGm (tile 原始区域，紧凑写回 GM，无需 UB 对齐)
        uint64_t outputBufElems = (uint64_t)splitSize_ * innerProdInGm_;

        // cast 类型 (fp16/bf16): dataBuf 按 PromoteT (4B) 分配，BulkCast 后存 F32；
        // outputBuf 仍按 T 分配（scatter 前在 VF 内 F32→B16 转换）
        if constexpr (kNeedCast) {
            dataBufBytes_ = CeilAlign(dataBufElems * (uint64_t)sizeof(PromoteT), (uint64_t)BLOCK_SIZE);
            outputBufBytes_ = CeilAlign(outputBufElems * (uint64_t)sizeof(T), (uint64_t)BLOCK_SIZE);
        } else {
            dataBufBytes_ = CeilAlign(dataBufElems * (uint64_t)sizeof(T), (uint64_t)BLOCK_SIZE);
            outputBufBytes_ = CeilAlign(outputBufElems * (uint64_t)sizeof(T), (uint64_t)BLOCK_SIZE);
        }

        pipe_->InitBuffer(dataBuf_, dataBufBytes_);
        pipe_->InitBuffer(outputBuf_, outputBufBytes_);
    }

    // ---------------- GM 紧凑 stride helper ----------------
    // GM 数据按 outputShape 紧凑布局，**不做 32B 对齐**（grad_out 侧）。
    // 用于 LoadGradOutRegion 的 GM 偏移与 LoopMode srcStride。
    __aicore__ inline uint64_t GmStride(uint32_t axis) const
    {
        uint64_t s = 1;
        for (uint32_t k = axis + 1; k < DIM_NUM; k++) {
            s *= outputShape_[k];
        }
        return s;
    }

    // ---------------- dataBuf 内 axis 切片间步长（元素数）----------------
    //   axis ∈ [0, SPLIT_AXIS)      —— 按 extent 紧密排布
    //   axis ∈ [SPLIT_AXIS, N-1]    —— 按 strideAligned_（中间维全量 outputShape + 尾轴 32B 对齐）
    //   axis < SPLIT_AXIS：stride = innerProdUb_ × ∏_{j=axis+1..SPLIT_AXIS} extent[j]
    //   axis ≥ SPLIT_AXIS：stride = strideAligned_[axis]
    __aicore__ inline uint64_t SliceStrideInBuf(uint32_t axis, const uint32_t* loadStart, const uint32_t* loadEnd) const
    {
        if (axis >= DIM_NUM)
            return 1;
        if (axis >= SPLIT_AXIS) {
            return strideAligned_[axis];
        }
        uint64_t s = innerProdUb_;
        for (uint32_t k = axis + 1; k <= SPLIT_AXIS; k++) {
            s *= (uint64_t)(loadEnd[k] - loadStart[k]);
        }
        return s;
    }

    // ---------------- Process（每核循环处理自己的 tile）----------------
    __aicore__ inline void Process()
    {
        uint64_t startIdx = (uint64_t)blockIdx_ * tilesPerCore_;
        if (startIdx >= splitCount_)
            return;
        uint64_t endIdx = startIdx + tilesPerCore_;
        if (endIdx > splitCount_)
            endIdx = splitCount_;

        for (uint64_t idx = startIdx; idx < endIdx; idx++) {
            ProcessOneTile(idx);
        }
    }

private:
    // ---------------- 单 tile 处理 ----------------
    __aicore__ inline void ProcessOneTile(uint64_t flatIdx)
    {
        // (1) 解码 flat idx → (outerCoords, tileInAxis)
        uint64_t outerCombo = flatIdx / splitCountAxis_;
        uint32_t tileInAxis = (uint32_t)(flatIdx % splitCountAxis_);

        uint32_t outerCoords[MAX_DIMS] = {0};
        uint64_t cur = outerCombo;
        for (int k = (int)SPLIT_AXIS - 1; k >= 0; k--) {
            outerCoords[k] = (uint32_t)(cur % inputShape_[k]);
            cur = cur / inputShape_[k];
        }

        uint32_t tileStart = tileInAxis * splitSize_;
        uint32_t tileEnd = tileStart + splitSize_;
        if (tileEnd > (uint32_t)inputShape_[SPLIT_AXIS])
            tileEnd = (uint32_t)inputShape_[SPLIT_AXIS];
        uint32_t tileLen = tileEnd - tileStart;
        bool firstTile = (tileStart == 0);
        bool lastTile = (tileEnd == (uint32_t)inputShape_[SPLIT_AXIS]);

        // (2) 计算每个轴的 load 区间
        uint32_t loadStart[MAX_DIMS] = {0};
        uint32_t loadEnd[MAX_DIMS] = {0};
        ComputeLoadRange(outerCoords, tileStart, tileEnd, firstTile, lastTile, loadStart, loadEnd);

        // (3) 一次性 DataCopyPad 搬入 grad_out → dataBuf（含 padding）
        LoadGradOutRegion(loadStart, loadEnd);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        // (3b) cast 类型：dataBuf B16 → F32 批量转换，后续累加/gather 均以 F32 进行
        if constexpr (kNeedCast) {
            BulkCastB16ToF32(dataBufElems_);
        }

        // (4) 累加 axis = 0 .. N-2 padding（kNeedCast 时以 F32 直接累加）
        AccumNonLastAxis(outerCoords, firstTile, lastTile, loadStart, loadEnd);

        // (5) axis = N-1 累加 + 提取原始区域 → outputBuf
        GatherToOutputBuf(outerCoords, tileStart, tileLen, firstTile, lastTile, loadStart, loadEnd);

        // (6) outputBuf → grad_in GM
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        OutputTileToGm(outerCoords, tileStart, tileLen);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    // ---------------- BulkCast B16→F32（仅 kNeedCast）----------------
    __aicore__ inline void BulkCastB16ToF32(uint64_t totalElems)
    {
        uint32_t VL_F32 = VREG_BYTES / sizeof(PromoteT); // 64
        uint32_t FULL = VL_F32 * 2;                      // 128 B16 = 2×64 F32
        uint16_t loops = CeilDiv((uint32_t)totalElems, FULL);

        LocalTensor<T> dataBuf = dataBuf_.Get<T>();
        auto* b16Ptr = reinterpret_cast<__ubuf__ T*>(dataBuf.GetPhyAddr());
        auto* f32Ptr = reinterpret_cast<__ubuf__ PromoteT*>(dataBuf.GetPhyAddr());

        uint32_t tail16 = totalElems % FULL;
        if (tail16 == 0 && totalElems > 0)
            tail16 = FULL;

        uint16_t hasTail = (tail16 > 0 && tail16 < FULL) ? (uint16_t)1 : (uint16_t)0;
        uint16_t nonTailCnt = loops - hasTail;
        uint32_t tailLo = tail16;
        if (tailLo > VL_F32)
            tailLo = VL_F32;
        uint32_t tailHi = (tail16 > VL_F32) ? (uint32_t)(tail16 - VL_F32) : 0;

        __VEC_SCOPE__
        {
            Reg::MaskReg maskB16, maskLo, maskHi;
            Reg::RegTensor<T> vregB16;
            Reg::RegTensor<PromoteT> vregF1, vregF2, vregF32Lo, vregF32Hi;

            // Step 1: tail (forward-last), hasTail=0 skip
            for (uint16_t t = 0; t < hasTail; t++) {
                uint32_t off = (uint32_t)(loops - 1) * FULL;
                maskB16 = Reg::UpdateMask<T>(tail16);
                maskLo = Reg::UpdateMask<PromoteT>(tailLo);
                maskHi = Reg::UpdateMask<PromoteT>(tailHi);
                Reg::DataCopy(vregB16, b16Ptr + off);
                Reg::Cast<PromoteT, T, CAST_TRAIT_PROMOTE_ZERO>(vregF1, vregB16, maskB16);
                Reg::Cast<PromoteT, T, CAST_TRAIT_PROMOTE_ONE>(vregF2, vregB16, maskB16);
                Reg::Interleave(vregF32Lo, vregF32Hi, vregF1, vregF2);
                Reg::DataCopy(f32Ptr + off, vregF32Lo, maskLo);
                Reg::DataCopy(f32Ptr + off + VL_F32, vregF32Hi, maskHi);
            }
            // Step 2: non-tail chunks backward (loops-2 ... 0)
            maskB16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < nonTailCnt; i++) {
                uint32_t off = (uint32_t)(loops - 1 - i - hasTail) * FULL;

                Reg::DataCopy(vregB16, b16Ptr + off);
                Reg::Cast<PromoteT, T, CAST_TRAIT_PROMOTE_ZERO>(vregF1, vregB16, maskB16);
                Reg::Cast<PromoteT, T, CAST_TRAIT_PROMOTE_ONE>(vregF2, vregB16, maskB16);
                Reg::Interleave(vregF32Lo, vregF32Hi, vregF1, vregF2);
                Reg::DataCopy(f32Ptr + off, vregF32Lo, maskB16);
                Reg::DataCopy(f32Ptr + off + VL_F32, vregF32Hi, maskB16);
            }
        }
    }

    // ---------------- 计算每个轴的 load 区间 ----------------
    __aicore__ inline void ComputeLoadRange(const uint32_t* outerCoords, uint32_t tileStart, uint32_t tileEnd,
                                            bool firstTile, bool lastTile, uint32_t* loadStart, uint32_t* loadEnd)
    {
        for (uint32_t k = 0; k < SPLIT_AXIS; k++) {
            bool leftEdge = (outerCoords[k] == 0) && (leftPad_[k] > 0);
            bool rightEdge = (outerCoords[k] == (uint32_t)inputShape_[k] - 1) && (rightPad_[k] > 0);
            if (leftEdge && rightEdge) {
                // inputShape[k]==1 且两端都有 padding：必须同时涵盖左右 padding 区间
                loadStart[k] = 0;
                loadEnd[k] = (uint32_t)outputShape_[k];
            } else if (leftEdge) {
                loadStart[k] = 0;
                loadEnd[k] = (uint32_t)leftPad_[k] + 1;
            } else if (rightEdge) {
                loadStart[k] = outerCoords[k] + (uint32_t)leftPad_[k];
                loadEnd[k] = (uint32_t)outputShape_[k];
            } else {
                loadStart[k] = outerCoords[k] + (uint32_t)leftPad_[k];
                loadEnd[k] = loadStart[k] + 1;
            }
        }
        loadStart[SPLIT_AXIS] = firstTile ? 0 : (tileStart + (uint32_t)leftPad_[SPLIT_AXIS]);
        loadEnd[SPLIT_AXIS] = lastTile ? (uint32_t)outputShape_[SPLIT_AXIS] :
                                         (tileEnd + (uint32_t)leftPad_[SPLIT_AXIS]);
        for (uint32_t k = SPLIT_AXIS + 1; k < DIM_NUM; k++) {
            loadStart[k] = 0;
            loadEnd[k] = (uint32_t)outputShape_[k];
        }
    }

    // ---------------- DataCopyPad 搬入多维矩形区域 ----------------
    //   - GM 数据按 outputShape 紧凑布局，GM 偏移用 GmStride()（不做 32B 对齐）。
    //   - UB 内尾轴（axis = N-1）按 32B 对齐布局，行间 stride = strideAligned_[N-2]，
    //     DataCopyPad 通过 dstStride（单位为 32B block）让 dst 自动按行对齐。
    //   - 搬入最小单位 = 1 行（尾轴 outputShape[N-1] 个元素，紧凑 blockLen），通过
    //     blockCount 一次搬入 "extent[SPLIT_AXIS] × ∏ outputShape[mid] " 行，
    //     消去 SPLIT_AXIS 和中间维（loadStart=0..outputShape[k]）的循环。
    //   - 在此基础上通过 LoopModeParams.loop1 / loop2 自动迭代外层轴 SPLIT_AXIS-1 与
    //     SPLIT_AXIS-2（如果存在），再外层轴 0..SPLIT_AXIS-3 仍需手动迭代。
    //   - dataBuf 布局：外层轴 0..SPLIT_AXIS-1 按 extent 紧密排布（对应 dataBuf 容量
    //     worstFactor × maxSliceAxis × innerProdUb 的分配方式），SPLIT_AXIS 与中间维按
    //     numRowsInner 行连续打包，尾轴按 strideAligned_[N-2] 32B 对齐。
    __aicore__ inline void LoadGradOutRegion(const uint32_t* loadStart, const uint32_t* loadEnd)
    {
        LocalTensor<T> dataLocal = dataBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // SPLIT_AXIS < DIM_NUM-1 ⇒ DIM_NUM ≥ 2
        if constexpr (DIM_NUM >= 2) {
            const uint32_t lastDim = (uint32_t)outputShape_[DIM_NUM - 1];
            const uint32_t rowBytesGm = lastDim * sizeof(T);
            const uint32_t rowBytesUb = (uint32_t)(strideAligned_[DIM_NUM - 2] * sizeof(T));
            const uint32_t dstStrideBlocks = (rowBytesUb - rowBytesGm) / BLOCK_SIZE;

            // 内层行数 = extent[SPLIT_AXIS] × ∏ outputShape[SPLIT_AXIS+1..N-2]
            uint64_t numRowsInner = loadEnd[SPLIT_AXIS] - loadStart[SPLIT_AXIS];
            for (uint32_t k = SPLIT_AXIS + 1; k <= (uint32_t)DIM_NUM - DIM_TWO; k++) {
                numRowsInner *= outputShape_[k];
            }

            DataCopyExtParams copyInParams;
            copyInParams.blockCount = (uint16_t)numRowsInner;
            copyInParams.blockLen = rowBytesGm;
            copyInParams.srcStride = 0; // GM 紧凑：相邻行 GM 间隔即 blockLen，"间距 - blockLen" = 0
            copyInParams.dstStride = dstStrideBlocks; // UB 行间 32B 对齐补齐

            if constexpr (SPLIT_AXIS == 0) {
                // 无外层 axis：单次 DataCopyPad
                const uint64_t gmOff = (uint64_t)loadStart[SPLIT_AXIS] * GmStride(SPLIT_AXIS);
                DataCopyPad(dataLocal, gradOutGm_[gmOff], copyInParams, padParams);
                return;
            }

            // UB 内"一次 DataCopyPad 内层块"字节数 = blockCount × rowBytesUb（含行 32B 对齐补齐）
            // 也等于 extent[SPLIT_AXIS] × innerProdUb_ × sizeof(T)
            const uint64_t innerBlockBytes = numRowsInner * (uint64_t)rowBytesUb;

            // LoopModeParams：loop1 = axis SPLIT_AXIS-1，loop2 = axis SPLIT_AXIS-2
            // src stride 用 GM 紧凑跨度（GmStride × sizeof(T)），dst stride 用 packed-extent UB 步长。
            LoopModeParams loopParams;
            loopParams.loop1Size = 1;
            loopParams.loop2Size = 1;
            bool useLoopParams = false;
            uint32_t extent1 = 1;
            uint32_t extent2 = 1;

            if constexpr (SPLIT_AXIS >= 1) {
                extent1 = loadEnd[SPLIT_AXIS - 1] - loadStart[SPLIT_AXIS - 1];
                loopParams.loop1Size = extent1;
                loopParams.loop1SrcStride = (uint32_t)(GmStride(SPLIT_AXIS - 1) * sizeof(T));
                loopParams.loop1DstStride = (uint32_t)innerBlockBytes;
                useLoopParams = useLoopParams || (extent1 > 1);
            }
            if constexpr (SPLIT_AXIS >= 2) {
                extent2 = loadEnd[SPLIT_AXIS - NUM_TWO] - loadStart[SPLIT_AXIS - NUM_TWO];
                loopParams.loop2Size = extent2;
                loopParams.loop2SrcStride = (uint32_t)(GmStride(SPLIT_AXIS - NUM_TWO) * sizeof(T));
                loopParams.loop2DstStride = (uint32_t)((uint64_t)extent1 * innerBlockBytes);
                useLoopParams = useLoopParams || (extent2 > 1);
            }

            // 一次手动外层迭代覆盖 inner × loop1 × loop2 个 packed-extent 单元
            const uint64_t outerStepBytes = (uint64_t)extent1 * extent2 * innerBlockBytes;

            constexpr int kManualOuterStart = (int)SPLIT_AXIS - 3;
            if constexpr (kManualOuterStart < 0) {
                // SPLIT_AXIS ∈ {1, 2}：单次 DataCopyPad 就能覆盖全部外层
                uint64_t gmOff = (uint64_t)loadStart[SPLIT_AXIS] * GmStride(SPLIT_AXIS);
                for (uint32_t k = 0; k < SPLIT_AXIS; k++) {
                    gmOff += (uint64_t)loadStart[k] * GmStride(k);
                }
                if (useLoopParams) {
                    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                    DataCopyPad(dataLocal, gradOutGm_[gmOff], copyInParams, padParams);
                    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
                } else {
                    DataCopyPad(dataLocal, gradOutGm_[gmOff], copyInParams, padParams);
                }
                return;
            }

            // 手动外层迭代 axes 0..SPLIT_AXIS-3（仅 SPLIT_AXIS ≥ 3 时进入）
            uint32_t coords[MAX_DIMS] = {0};
            for (int k = 0; k <= kManualOuterStart; k++)
                coords[k] = loadStart[k];

            uint64_t dstOffBytes = 0;
            while (true) {
                uint64_t gmOff = (uint64_t)loadStart[SPLIT_AXIS] * GmStride(SPLIT_AXIS);
                if constexpr (SPLIT_AXIS >= 1) {
                    gmOff += (uint64_t)loadStart[SPLIT_AXIS - 1] * GmStride(SPLIT_AXIS - 1);
                }
                if constexpr (SPLIT_AXIS >= 2) {
                    gmOff += (uint64_t)loadStart[SPLIT_AXIS - NUM_TWO] * GmStride(SPLIT_AXIS - NUM_TWO);
                }
                for (int k = 0; k <= kManualOuterStart; k++) {
                    gmOff += (uint64_t)coords[k] * GmStride(k);
                }

                const uint64_t dstOffElems = dstOffBytes / sizeof(T);
                if (useLoopParams) {
                    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                    DataCopyPad(dataLocal[dstOffElems], gradOutGm_[gmOff], copyInParams, padParams);
                    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
                } else {
                    DataCopyPad(dataLocal[dstOffElems], gradOutGm_[gmOff], copyInParams, padParams);
                }
                dstOffBytes += outerStepBytes;

                int k = kManualOuterStart;
                for (; k >= 0; k--) {
                    coords[k]++;
                    if (coords[k] < loadEnd[k])
                        break;
                    coords[k] = loadStart[k];
                }
                if (k < 0)
                    break;
            }
        } // if constexpr (DIM_NUM >= 2)
    }

    // ---------------- 累加 axis = 0 .. N-2 padding ----------------
    // 处理顺序：axis = 0, 1, ..., N-2。每个 axis 把自身 padding slabs 累加到 boundary slab。
    // 切片大小 sliceElems = SliceStrideInBuf(axis)（dataBuf 内 axis 一个 slab 的连续元素数）。
    // axis 自身 slab 索引（lDstRel/lSrcStart/rDstRel/rSrcStart）三种情况：
    //   axis < SPLIT_AXIS：仅在 outerCoords[axis] 在边缘时存在 src slabs。
    //   axis == SPLIT_AXIS：firstTile/lastTile 才有 src slabs。
    //   axis > SPLIT_AXIS（仍 < N-1）：在 outputShape 范围内必定存在两侧 pad slabs。
    // 外层位置（j < axis）只迭代"非 padding"区间，跳过 axis-j padding slabs。理由：
    //   axis j 已在更早的 pass 把 padding slabs 累加进 axis-j 的 boundary 了；这些 slabs
    //   之后写入"已消费"区域，再让 axis K 重跑只是浪费 vec 时间（写出位置落在某个
    //   axis-j padding 区域，不会出现在最终输出）。各 j 的有效区间：
    //     j <  SPLIT_AXIS：只剩 boundary slot（左边缘=pL_j；右边缘=0；中间=0）
    //     j == SPLIT_AXIS：当前 tile 的非 pad 区段，长度 = tileLen
    //                      （firstTile 起点偏 pL_{SPLIT_AXIS}，lastTile 终点提前 pR）
    //     j >  SPLIT_AXIS：[pL_j, pL_j + inputShape[j])，长度 = inputShape[j]
    __aicore__ inline void AccumNonLastAxis(const uint32_t* outerCoords, bool firstTile, bool lastTile,
                                            const uint32_t* loadStart, const uint32_t* loadEnd)
    {
        // BulkCast 后 kNeedCast 时 dataBuf 为 F32，用 PromoteT 视图
        if constexpr (kNeedCast) {
            LocalTensor<PromoteT> dataF = dataBuf_.Get<PromoteT>();
            AccumNonLastAxisImpl(dataF, outerCoords, firstTile, lastTile, loadStart, loadEnd);
        } else {
            LocalTensor<T> data = dataBuf_.Get<T>();
            AccumNonLastAxisImpl(data, outerCoords, firstTile, lastTile, loadStart, loadEnd);
        }
    }

    template <typename DType>
    __aicore__ inline void AccumNonLastAxisImpl(LocalTensor<DType>& data, const uint32_t* outerCoords, bool firstTile,
                                                bool lastTile, const uint32_t* loadStart, const uint32_t* loadEnd)
    {
        // 当前 tile 切分轴上"有效（非 pad）"段长度。三种情况都退化为 tileLen：
        //   firstTile && lastTile：extent = outputShape，两端扣除 pL+pR
        //   firstTile / lastTile：extent = tileLen + 单端 pad
        //   middle：extent = tileLen
        const uint32_t splitExtent = loadEnd[SPLIT_AXIS] - loadStart[SPLIT_AXIS];
        const uint32_t splitPadL = firstTile ? (uint32_t)leftPad_[SPLIT_AXIS] : 0u;
        const uint32_t splitPadR = lastTile ? (uint32_t)rightPad_[SPLIT_AXIS] : 0u;
        const uint32_t tileLen = splitExtent - splitPadL - splitPadR;

        for (uint32_t axis = 0; axis < DIM_NUM - 1; axis++) {
            if (leftPad_[axis] == 0 && rightPad_[axis] == 0)
                continue;

            bool doLeft = false;
            bool doRight = false;
            uint32_t totalSlices = loadEnd[axis] - loadStart[axis];

            uint64_t sliceStrideElems = SliceStrideInBuf(axis, loadStart, loadEnd);
            uint64_t sliceElems = sliceStrideElems;

            // 决定 axis 自身的 src/dst 切片范围（slab 索引相对当前 axis 的起点）
            uint32_t lSrcStart = 0, lSrcCnt = 0, lDstRel = 0;
            uint32_t rSrcStart = 0, rSrcCnt = 0, rDstRel = 0;

            if (axis < SPLIT_AXIS) {
                bool leftEdge = (outerCoords[axis] == 0) && (leftPad_[axis] > 0);
                bool rightEdge = (outerCoords[axis] == (uint32_t)inputShape_[axis] - 1) && (rightPad_[axis] > 0);
                if (leftEdge) {
                    lSrcStart = 0;
                    lSrcCnt = (uint32_t)leftPad_[axis];
                    lDstRel = (uint32_t)leftPad_[axis];
                    doLeft = (lSrcCnt > 0);
                }
                if (rightEdge) {
                    if (leftEdge) {
                        // inputShape[axis]==1 且两端都有 padding：右 padding 累加到同一个 interior 位置
                        rDstRel = (uint32_t)leftPad_[axis];
                        rSrcStart = (uint32_t)leftPad_[axis] + 1;
                        rSrcCnt = (uint32_t)rightPad_[axis];
                    } else {
                        rDstRel = 0;
                        rSrcStart = 1;
                        rSrcCnt = totalSlices - 1;
                    }
                    doRight = (rSrcCnt > 0);
                }
            } else if (axis == SPLIT_AXIS) {
                if (firstTile && leftPad_[axis] > 0) {
                    lSrcStart = 0;
                    lSrcCnt = (uint32_t)leftPad_[axis];
                    lDstRel = (uint32_t)leftPad_[axis];
                    doLeft = true;
                }
                if (lastTile && rightPad_[axis] > 0) {
                    rDstRel = totalSlices - 1 - (uint32_t)rightPad_[axis];
                    rSrcStart = rDstRel + 1;
                    rSrcCnt = (uint32_t)rightPad_[axis];
                    doRight = true;
                }
            } else {
                if (leftPad_[axis] > 0) {
                    lSrcStart = 0;
                    lSrcCnt = (uint32_t)leftPad_[axis];
                    lDstRel = (uint32_t)leftPad_[axis];
                    doLeft = true;
                }
                if (rightPad_[axis] > 0) {
                    rDstRel = (uint32_t)((uint32_t)inputShape_[axis] - 1 + leftPad_[axis]);
                    rSrcStart = rDstRel + 1;
                    rSrcCnt = (uint32_t)rightPad_[axis];
                    doRight = true;
                }
            }

            if (!doLeft && !doRight)
                continue;

            // ============ 外层"有效位置"迭代 ============
            // 固定部分（j < min(axis, SPLIT_AXIS)）：每个 j 都只剩 boundary slot。
            //   左边缘 (outerCoords[j]==0 && pL_j>0)：dataBuf c_j = pL_j
            //   右边缘 / 中间：dataBuf c_j = 0
            uint64_t baseFixed = 0;
            const uint32_t fixedEnd = (axis < (uint32_t)SPLIT_AXIS) ? axis : (uint32_t)SPLIT_AXIS;
            for (uint32_t j = 0; j < fixedEnd; j++) {
                uint64_t bndCoord = 0;
                if (outerCoords[j] == 0 && leftPad_[j] > 0) {
                    bndCoord = (uint64_t)leftPad_[j];
                }
                baseFixed += bndCoord * SliceStrideInBuf(j, loadStart, loadEnd);
            }

            // 变动部分（j ∈ [SPLIT_AXIS, axis)）：仅 axis > SPLIT_AXIS 时非空。
            //   j == SPLIT_AXIS：iterStart = firstTile ? pL_{SPLIT_AXIS} : 0，count = tileLen
            //   j >  SPLIT_AXIS：iterStart = pL_j，count = inputShape[j]
            uint32_t numVar = (axis > (uint32_t)SPLIT_AXIS) ? (axis - (uint32_t)SPLIT_AXIS) : 0u;
            uint64_t varStart[MAX_DIMS] = {0};
            uint64_t varCount[MAX_DIMS] = {0};
            uint64_t varStride[MAX_DIMS] = {0};
            for (uint32_t v = 0; v < numVar; v++) {
                uint32_t j = (uint32_t)SPLIT_AXIS + v;
                if (j == (uint32_t)SPLIT_AXIS) {
                    varStart[v] = (uint64_t)splitPadL;
                    varCount[v] = (uint64_t)tileLen;
                } else {
                    varStart[v] = (uint64_t)leftPad_[j];
                    varCount[v] = (uint64_t)inputShape_[j];
                }
                varStride[v] = SliceStrideInBuf(j, loadStart, loadEnd);
            }

            uint64_t outerCount = 1;
            for (uint32_t v = 0; v < numVar; v++)
                outerCount *= varCount[v];
            if (outerCount == 0)
                continue;

            for (uint64_t o = 0; o < outerCount; o++) {
                // 把扁平 o 解码为 (idx_{SPLIT_AXIS}, ..., idx_{axis-1})，innermost 变化最快
                uint64_t baseElems = baseFixed;
                uint64_t rem = o;
                for (int v = (int)numVar - 1; v >= 0; v--) {
                    uint64_t idx = rem % varCount[v];
                    rem /= varCount[v];
                    baseElems += (varStart[v] + idx) * varStride[v];
                }

                if (doLeft) {
                    AccumSlicesVf(data, baseElems + (uint64_t)lDstRel * sliceStrideElems,
                                  baseElems + (uint64_t)lSrcStart * sliceStrideElems, sliceStrideElems, lSrcCnt,
                                  sliceElems);
                }
                if (doRight) {
                    AccumSlicesVf(data, baseElems + (uint64_t)rDstRel * sliceStrideElems,
                                  baseElems + (uint64_t)rSrcStart * sliceStrideElems, sliceStrideElems, rSrcCnt,
                                  sliceElems);
                }
            }
        }
    }

    // 在 vec_scope 内执行：data[dstOffset, dstOffset+sliceElems) += Σ data[srcStart + j*srcStride, ...)
    // DType = T（非 cast）或 PromoteT（kNeedCast, BulkCast 后 dataBuf 已为 F32）
    template <typename DType>
    __aicore__ inline void AccumSlicesVf(LocalTensor<DType>& data, uint64_t dstOffsetElems, uint64_t srcStartElems,
                                         uint64_t srcStrideElems, uint32_t srcCnt, uint64_t sliceElems)
    {
        if (srcCnt == 0 || sliceElems == 0)
            return;

        auto baseAddr = reinterpret_cast<__local_mem__ DType*>(data.GetPhyAddr());
        constexpr uint32_t VL = VREG_BYTES / sizeof(DType);
        const uint16_t loopCount = (uint16_t)CeilDiv((uint32_t)sliceElems, VL);

        __VEC_SCOPE__
        {
            uint32_t remain = (uint32_t)sliceElems;
            Reg::MaskReg mask;
            Reg::RegTensor<DType> vregDst, vregSrc;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask = Reg::UpdateMask<DType>(remain);
                Reg::DataCopy(vregDst, baseAddr + dstOffsetElems + i * VL);
                for (uint16_t j = 0; j < (uint16_t)srcCnt; j++) {
                    Reg::DataCopy(vregSrc, baseAddr + srcStartElems + j * srcStrideElems + i * VL);
                    Reg::Add(vregDst, vregDst, vregSrc, mask);
                }
                Reg::DataCopy(baseAddr + dstOffsetElems + i * VL, vregDst, mask);
            }
        }
    }

    // ---------------- axis=N-1 累加 + 提取原始区域 → outputBuf ----------------
    //   - 非 cast 类型（FP32/INT32/...）：使用 Gather/Scatter 跨行并行（设计文档 7.4.2）
    //     index 类型需匹配数据宽度：B32→uint32_t（Arange 用 int32_t 后 reinterpret cast）
    //   - cast 类型（FP16/BF16）：Gather/Scatter 跨行并行 + 在 vec_scope 内做 cast→Add→cast
    //     index 类型 = uint16_t（B16 gather/scatter 强制 16bit）；
    //     一次处理 VL = VREG_BYTES/sizeof(PromoteT) = 64 行（PromoteT lane 数）。
    //     Phase B（中间，无累加）跳过 cast/Add，直接 B16 gather → scatter。
    //   每个原始 group g 对应 dataBuf 中 row offset =
    //     firstRowElems + Σ_{k∈[SPLIT_AXIS, N-2]} idx_k(g) × strideAligned_[k]
    //   其中 idx_{SPLIT_AXIS}(g) ∈ [0, tileLen)，idx_{k>SPLIT_AXIS}(g) ∈ [0, inputShape_[k])。
    //   "前段合并"优化：后 5 维以前的 axes（k < kPadAxisStart）若 padding=0，
    //   则 outputShape_[k] = inputShape_[k]，strideAligned_ 链与 inputShape 一致，可合并为
    //   1 个虚拟 axis。最终 kEffAxes ∈ [1, 5]，effExtents/effStrides 按"内→外"排列。
    __aicore__ inline void GatherToOutputBuf(const uint32_t* outerCoords, uint32_t tileStart, uint32_t tileLen,
                                             bool firstTile, bool lastTile, const uint32_t* loadStart,
                                             const uint32_t* loadEnd)
    {
        LocalTensor<T> data = dataBuf_.Get<T>();
        LocalTensor<T> output = outputBuf_.Get<T>();

        const uint32_t dN1 = (uint32_t)inputShape_[DIM_NUM - 1];
        const int32_t pL = leftPad_[DIM_NUM - 1];
        const int32_t pR = rightPad_[DIM_NUM - 1];

        // 第一个原始行在 dataBuf 中的元素偏移
        uint32_t firstOrigCoords[MAX_DIMS] = {0};
        for (uint32_t k = 0; k < SPLIT_AXIS; k++)
            firstOrigCoords[k] = outerCoords[k];
        firstOrigCoords[SPLIT_AXIS] = tileStart;
        for (uint32_t k = SPLIT_AXIS + 1; k < DIM_NUM - 1; k++)
            firstOrigCoords[k] = 0;
        const uint64_t firstRowElems = ComputeRowOffsetInBuf(firstOrigCoords, loadStart, loadEnd);

        // ---- 多维"原始行 group"轴信息（按"内→外"排列）----
        // 后 5 维起始 axis（DIM_NUM<=5 时为 0；DIM_NUM>5 时为 DIM_NUM-5）
        constexpr int8_t kPadAxisStart = (DIM_NUM > MAX_PAD_DIMS) ? (int8_t)(DIM_NUM - MAX_PAD_DIMS) : (int8_t)0;
        constexpr bool kHasFrontMerge = ((int8_t)SPLIT_AXIS < kPadAxisStart);
        constexpr int8_t kRearStart = kHasFrontMerge ? kPadAxisStart : (int8_t)SPLIT_AXIS;
        // 后段 axes 数：从 kRearStart 到 N-2（含），共 (N-2) - kRearStart + 1 = N-1-kRearStart 个
        constexpr int8_t kRearAxes = (int8_t)(DIM_NUM - 1) - kRearStart;
        constexpr uint8_t kEffAxes = (uint8_t)kRearAxes + (kHasFrontMerge ? 1u : 0u);

        // 构造 effExtents / effStrides（内→外排列：[0] = innermost = axis N-2）
        uint32_t effExtents[MAX_PAD_DIMS] = {1, 1, 1, 1, 1};
        uint32_t effStrides[MAX_PAD_DIMS] = {0, 0, 0, 0, 0};
        // Rear: axes [kRearStart, N-2]，innermost-first
        for (int8_t i = 0; i < kRearAxes; i++) {
            int8_t axis = (int8_t)(DIM_NUM - 2) - i; // axis N-2, N-3, ..., kRearStart
            uint32_t ext;
            if (axis == (int8_t)SPLIT_AXIS) {
                ext = tileLen; // 仅 SPLIT_AXIS 等于 kRearStart 时（无 front merge），取 tileLen
            } else {
                ext = (uint32_t)inputShape_[axis];
            }
            effExtents[i] = ext;
            effStrides[i] = (uint32_t)strideAligned_[axis];
        }
        // Front merged virtual axis（仅 DIM_NUM>5 且 SPLIT_AXIS<kPadAxisStart 时存在）
        if constexpr (kHasFrontMerge) {
            uint32_t mergedExt = tileLen; // SPLIT_AXIS 的 extent 用 tileLen 替换
            for (int8_t k = (int8_t)SPLIT_AXIS + 1; k <= (int8_t)(kPadAxisStart - 1); k++) {
                mergedExt *= (uint32_t)inputShape_[k];
            }
            effExtents[kRearAxes] = mergedExt;
            effStrides[kRearAxes] = (uint32_t)strideAligned_[kPadAxisStart - 1];
        }

        // 原始行数（除最低维外的位置数）= ∏ effExtents
        uint32_t totalGroups = 1;
        for (uint8_t i = 0; i < kEffAxes; i++)
            totalGroups *= effExtents[i];

        if constexpr (kNeedCast) {
            GatherToOutputBufF32Path(totalGroups, firstRowElems, effExtents, effStrides, dN1, pL, pR);
        } else {
            GatherToOutputBufGatherPath(data, output, totalGroups, firstRowElems, effExtents, effStrides, dN1, pL, pR);
        }
    }

    // ---------------- F32 cast 路径（新方案）：gather F32 → accumulate F32 → cast F32→B16 → scatter B16
    // BulkCast 后 dataBuf 已为 F32，所有 gather/累加 在 F32 下进行，scatter 前做一次 F32→B16 cast。
    __aicore__ inline void GatherToOutputBufF32Path(uint32_t totalGroups, uint64_t firstRowElems,
                                                    const uint32_t* effExtents, const uint32_t* effStrides,
                                                    uint32_t dN1, int32_t pL, int32_t pR)
    {
        using Idx32 = uint32_t; // F32 gather: 64 lanes/256B
        using Idx16 = uint16_t; // B16 scatter: 128 lanes/256B
        using Rng32 = int32_t;
        using Rng16 = int16_t;

        uint32_t VL_F32 = VREG_BYTES / sizeof(PromoteT); // 64
        uint16_t numBatches = (totalGroups + VL_F32 - 1) / VL_F32;
        uint32_t totalGrp = totalGroups;
        const Rng32 baseE = (Rng32)firstRowElems;
        const Rng32 outRowE = (Rng32)dN1;
        const uint16_t pLu = (uint16_t)pL;
        const uint16_t pRu = (uint16_t)pR;
        const Rng32 rightPadStart = (Rng32)pL + (Rng32)dN1;
        const uint16_t midCount = (dN1 >= 3) ? (uint16_t)(dN1 - 2) : (uint16_t)0;
        const uint16_t iRight = (uint16_t)(dN1 - 1);
        const uint16_t phaseA_pLu = (dN1 <= 1) ? (uint16_t)0 : pLu;
        const uint16_t combPadCnt = (dN1 <= 1) ? pLu : (uint16_t)0;

        auto* dataAddr = reinterpret_cast<__ubuf__ PromoteT*>(dataBuf_.Get<T>().GetPhyAddr());
        auto* outputAddr = reinterpret_cast<__ubuf__ T*>(outputBuf_.Get<T>().GetPhyAddr());

        // 多维 base 索引常量（与 GatherToOutputBuf 对齐）
        constexpr int8_t kPadAxisStart = (DIM_NUM > MAX_PAD_DIMS) ? (int8_t)(DIM_NUM - MAX_PAD_DIMS) : (int8_t)0;
        constexpr bool kHasFrontMerge = ((int8_t)SPLIT_AXIS < kPadAxisStart);
        constexpr int8_t kRearStart = kHasFrontMerge ? kPadAxisStart : (int8_t)SPLIT_AXIS;
        constexpr uint8_t kEffAxes = (uint8_t)((int8_t)(DIM_NUM - 1) - kRearStart) + (kHasFrontMerge ? 1u : 0u);

        const Idx32 eE0 = (Idx32)effExtents[0];
        const Idx32 eE1 = (Idx32)effExtents[1];
        const Idx32 eE2 = (Idx32)effExtents[2];
        const Idx32 eE3 = (Idx32)effExtents[3];
        const Idx32 eS0 = (Idx32)effStrides[0];
        const Idx32 eS1 = (Idx32)effStrides[1];
        const Idx32 eS2 = (Idx32)effStrides[2];
        const Idx32 eS3 = (Idx32)effStrides[3];
        const Idx32 eS4 = (Idx32)effStrides[4];

        __VEC_SCOPE__
        {
            // F32 侧
            Reg::MaskReg maskF32;
            Reg::RegTensor<Rng32> tmpRange32;
            Reg::RegTensor<Idx32> baseIdx, quot, idxF32, tmpReg;
            Reg::RegTensor<Idx32> qNext, ik, dExt;
            Reg::RegTensor<PromoteT> vregF32, vregPadF32;

            // B16 侧 (scatter + cast)
            Reg::MaskReg maskB16, selMaskLo, selMaskHi, allMask;
            Reg::RegTensor<Rng16> tmpRange16, tmpHalf;
            Reg::RegTensor<Idx16> idxB16;
            Reg::RegTensor<T> vregB16Lo, vregB16Hi, vregScatter;

            for (uint16_t b = 0; b < numBatches; b++) {
                uint32_t remain = totalGrp * 2;
                maskF32 = Reg::UpdateMask<PromoteT>(totalGrp);
                maskB16 = Reg::UpdateMask<T>(remain);
                allMask = Reg::CreateMask<T, Reg::MaskPattern::ALLF>();
                Reg::MaskInterleave<T>(selMaskLo, selMaskHi, maskB16, allMask);

                // === shared baseIdx computation (Idx32 + maskF32) ===
                Reg::Arange(tmpRange32, (Rng32)0);
                quot = (Reg::RegTensor<Idx32>&)tmpRange32;
                Reg::Adds(quot, quot, (Idx32)((Rng32)b * (Rng32)VL_F32), maskF32);
                Reg::Duplicate(baseIdx, (Idx32)baseE, maskF32);
                if constexpr (kEffAxes >= 2) {
                    Reg::Duplicate(dExt, eE0, maskF32);
                    Reg::Div<Idx32>(qNext, quot, dExt, maskF32);
                    Reg::Muls(tmpReg, qNext, eE0, maskF32);
                    Reg::Sub<Idx32>(ik, quot, tmpReg, maskF32);
                    Reg::Muls(tmpReg, ik, eS0, maskF32);
                    Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                    Reg::Copy(quot, qNext);
                    if constexpr (kEffAxes >= 3) {
                        Reg::Duplicate(dExt, eE1, maskF32);
                        Reg::Div<Idx32>(qNext, quot, dExt, maskF32);
                        Reg::Muls(tmpReg, qNext, eE1, maskF32);
                        Reg::Sub<Idx32>(ik, quot, tmpReg, maskF32);
                        Reg::Muls(tmpReg, ik, eS1, maskF32);
                        Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                        Reg::Copy(quot, qNext);
                        if constexpr (kEffAxes >= 4) {
                            Reg::Duplicate(dExt, eE2, maskF32);
                            Reg::Div<Idx32>(qNext, quot, dExt, maskF32);
                            Reg::Muls(tmpReg, qNext, eE2, maskF32);
                            Reg::Sub<Idx32>(ik, quot, tmpReg, maskF32);
                            Reg::Muls(tmpReg, ik, eS2, maskF32);
                            Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                            Reg::Copy(quot, qNext);
                            if constexpr (kEffAxes >= 5) {
                                Reg::Duplicate(dExt, eE3, maskF32);
                                Reg::Div<Idx32>(qNext, quot, dExt, maskF32);
                                Reg::Muls(tmpReg, qNext, eE3, maskF32);
                                Reg::Sub<Idx32>(ik, quot, tmpReg, maskF32);
                                Reg::Muls(tmpReg, ik, eS3, maskF32);
                                Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                                Reg::Copy(quot, qNext);
                                Reg::Muls(tmpReg, quot, eS4, maskF32);
                                Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                            } else {
                                Reg::Muls(tmpReg, quot, eS3, maskF32);
                                Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                            }
                        } else {
                            Reg::Muls(tmpReg, quot, eS2, maskF32);
                            Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                        }
                    } else {
                        Reg::Muls(tmpReg, quot, eS1, maskF32);
                        Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                    }
                } else {
                    Reg::Muls(tmpReg, quot, eS0, maskF32);
                    Reg::Add<Idx32>(baseIdx, baseIdx, tmpReg, maskF32);
                }

                // ======== Phase A: 左边界 ========
                Reg::Adds(idxF32, baseIdx, (Idx32)pL, maskF32);
                Reg::DataCopyGather(vregF32, dataAddr, idxF32, maskF32);
                for (uint16_t k = 0; k < phaseA_pLu; k++) {
                    Reg::Adds(idxF32, baseIdx, (Idx32)k, maskF32);
                    Reg::DataCopyGather(vregPadF32, dataAddr, idxF32, maskF32);
                    Reg::Add(vregF32, vregF32, vregPadF32, maskF32);
                }
                // F32→B16 cast + Select
                Reg::Cast<T, PromoteT, CAST_TRAIT_DOWN_ZERO>(vregB16Lo, vregF32, maskB16);
                Reg::Cast<T, PromoteT, CAST_TRAIT_DOWN_ONE>(vregB16Hi, vregF32, maskB16);
                Reg::Select(vregScatter, vregB16Lo, vregB16Hi, selMaskLo);
                // scatter B16 to outputBuf (Idx16 + maskB16)
                Reg::Arange(tmpRange16, (Rng16)0);
                Reg::Duplicate(tmpHalf, (Rng16)2, maskB16);
                Reg::Div<Idx16>((Reg::RegTensor<Idx16>&)tmpRange16, (Reg::RegTensor<Idx16>&)tmpRange16,
                                (Reg::RegTensor<Idx16>&)tmpHalf, maskB16);
                Reg::Muls(idxB16, (Reg::RegTensor<Idx16>&)tmpRange16, (Idx16)dN1, maskB16);
                Reg::Adds(idxB16, idxB16, (Idx16)((Rng16)b * (Rng16)VL_F32 * (Rng16)dN1), maskB16);
                Reg::DataCopyScatter(outputAddr, vregScatter, idxB16, maskB16);

                // ======== Phase B: 中间 ========
                for (uint16_t i = 0; i < midCount; i++) {
                    const uint16_t iN1 = i + 1;
                    Reg::Adds(idxF32, baseIdx, (Idx32)((Rng32)iN1 + (Rng32)pL), maskF32);
                    Reg::DataCopyGather(vregF32, dataAddr, idxF32, maskF32);
                    Reg::Cast<T, PromoteT, CAST_TRAIT_DOWN_ZERO>(vregB16Lo, vregF32, maskB16);
                    Reg::Cast<T, PromoteT, CAST_TRAIT_DOWN_ONE>(vregB16Hi, vregF32, maskB16);
                    Reg::Select(vregScatter, vregB16Lo, vregB16Hi, selMaskLo);
                    Reg::Arange(tmpRange16, (Rng16)0);
                    Reg::Duplicate(tmpHalf, (Rng16)2, maskB16);
                    Reg::Div<Idx16>((Reg::RegTensor<Idx16>&)tmpRange16, (Reg::RegTensor<Idx16>&)tmpRange16,
                                    (Reg::RegTensor<Idx16>&)tmpHalf, maskB16);
                    Reg::Muls(idxB16, (Reg::RegTensor<Idx16>&)tmpRange16, (Idx16)dN1, maskB16);
                    Reg::Adds(idxB16, idxB16, (Idx16)((Rng16)b * (Rng16)VL_F32 * (Rng16)dN1 + (Rng16)iN1), maskB16);
                    Reg::DataCopyScatter(outputAddr, vregScatter, idxB16, maskB16);
                }

                // ======== Phase C: 右边界 ========
                Reg::Adds(idxF32, baseIdx, (Idx32)((Rng32)pL + (Rng32)iRight), maskF32);
                Reg::DataCopyGather(vregF32, dataAddr, idxF32, maskF32);
                for (uint16_t k = 0; k < combPadCnt; k++) {
                    Reg::Adds(idxF32, baseIdx, (Idx32)k, maskF32);
                    Reg::DataCopyGather(vregPadF32, dataAddr, idxF32, maskF32);
                    Reg::Add(vregF32, vregF32, vregPadF32, maskF32);
                }
                for (uint16_t k = 0; k < pRu; k++) {
                    Reg::Adds(idxF32, baseIdx, (Idx32)(rightPadStart + (Rng32)k), maskF32);
                    Reg::DataCopyGather(vregPadF32, dataAddr, idxF32, maskF32);
                    Reg::Add(vregF32, vregF32, vregPadF32, maskF32);
                }
                Reg::Cast<T, PromoteT, CAST_TRAIT_DOWN_ZERO>(vregB16Lo, vregF32, maskB16);
                Reg::Cast<T, PromoteT, CAST_TRAIT_DOWN_ONE>(vregB16Hi, vregF32, maskB16);
                Reg::Select(vregScatter, vregB16Lo, vregB16Hi, selMaskLo);
                Reg::Arange(tmpRange16, (Rng16)0);
                Reg::Duplicate(tmpHalf, (Rng16)2, maskB16);
                Reg::Div<Idx16>((Reg::RegTensor<Idx16>&)tmpRange16, (Reg::RegTensor<Idx16>&)tmpRange16,
                                (Reg::RegTensor<Idx16>&)tmpHalf, maskB16);
                Reg::Muls(idxB16, (Reg::RegTensor<Idx16>&)tmpRange16, (Idx16)dN1, maskB16);
                Reg::Adds(idxB16, idxB16, (Idx16)((Rng16)b * (Rng16)VL_F32 * (Rng16)dN1 + (Rng16)iRight), maskB16);
                Reg::DataCopyScatter(outputAddr, vregScatter, idxB16, maskB16);
            }
        }
    }

    // ---------------- 非 cast 路径：Gather/Scatter 跨行并行 ----------------
    // 位宽-IndexT-CastT 对应表（硬件约束，见 kernel_micro_datacopy_impl.h:62 / :157）：
    //   B8   sizeof(T)==1  Gather src→dst: b8 →b16; Scatter: b8 ; IndexT=uint16_t; RangeT=int16_t 。
    //                      → 需 reinterpret 到 RegTensor<CastT=uint16_t/int16_t>，再 Pack 回 B8
    //   B16  sizeof(T)==2  Gather src→dst: b16→b16; Scatter: b16; IndexT=uint16_t; RangeT=int16_t 。
    //   B32  sizeof(T)==4  Gather src→dst: b32→b32; Scatter: b32; IndexT=uint32_t; RangeT=int32_t 。
    //   B64  sizeof(T)==8  Gather src→dst: b64→b64; Scatter: b64; IndexT=uint32_t（前 32 数）; RangeT=int32_t 。
    // index 单位为**元素索引**（不是字节偏移），与 pad_v3/pad_scatter.h、roll/roll_gather_simd.h 一致。
    __aicore__ inline void GatherToOutputBufGatherPath(LocalTensor<T>& data, LocalTensor<T>& output,
                                                       uint32_t totalGroups, uint64_t firstRowElems,
                                                       const uint32_t* effExtents, const uint32_t* effStrides,
                                                       uint32_t dN1, int32_t pL, int32_t pR)
    {
        // 编译期分派 IndexT / RangeT / CastT
        using IndexT = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
        using RangeT = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
        // CastT：B8 时 Gather/Scatter 的 dst/src 寄存器需扩展到 B16（DataCopyGather 自动 src=b8→dst=b16）
        // 写法参考 conversion/pad_v3/op_kernel/arch35/pad_scatter.h（uint8_t→uint16_t, int8_t→int16_t）
        using CastT = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

        // index 单位为元素索引（不是字节偏移）
        const RangeT baseE = (RangeT)firstRowElems;
        const RangeT outRowE = (RangeT)dN1;
        constexpr uint32_t VL = VREG_BYTES / sizeof(T);
        // B8 (sizeof(T)=1): VL=256 but IndexT=uint16_t has only 128 lanes, so batch size
        // must be capped at IndexT lane count.  For B16/B32/B64, kBatchVL == VL.
        constexpr uint32_t kIdxVL = VREG_BYTES / sizeof(IndexT);
        constexpr uint32_t kBatchVL = (VL < kIdxVL) ? VL : kIdxVL;
        const uint16_t numBatches = (uint16_t)((totalGroups + kBatchVL - 1) / kBatchVL);
        const uint16_t pLu = (uint16_t)pL;
        const uint16_t pRu = (uint16_t)pR;
        const RangeT rightPadStart = (RangeT)pL + (RangeT)dN1;
        // 中间段 iN1 ∈ [1, dN1-1)，共 midCount 个；
        // 循环用 0-起始变量 i，内部 iN1 = i + 1（VF 规范禁止 for 起始非 0）。
        const uint16_t midCount = (dN1 >= 3) ? (uint16_t)(dN1 - 2) : (uint16_t)0;
        const uint16_t iRight = (uint16_t)(dN1 - 1);
        // dN1==1: left/right boundary are the same element; Phase C overwrites Phase A's scatter,
        // so skip Phase A left-padding loop and re-accumulate it in Phase C instead.
        const uint16_t phaseA_pLu = (dN1 <= 1) ? (uint16_t)0 : pLu;
        const uint16_t combPadCnt = (dN1 <= 1) ? pLu : (uint16_t)0;

        auto dataAddr = reinterpret_cast<__ubuf__ T*>(data.GetPhyAddr());
        auto outputAddr = reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr());

        // 多维 base 索引相关编译期常量（与 GatherToOutputBuf 对齐）
        constexpr int8_t kPadAxisStart = (DIM_NUM > MAX_PAD_DIMS) ? (int8_t)(DIM_NUM - MAX_PAD_DIMS) : (int8_t)0;
        constexpr bool kHasFrontMerge = ((int8_t)SPLIT_AXIS < kPadAxisStart);
        constexpr int8_t kRearStart = kHasFrontMerge ? kPadAxisStart : (int8_t)SPLIT_AXIS;
        constexpr uint8_t kEffAxes = (uint8_t)((int8_t)(DIM_NUM - 1) - kRearStart) + (kHasFrontMerge ? 1u : 0u);

        // 把 effExtents/effStrides 拷到局部 IndexT 域常量（避免 vec_scope 内访问 uint32_t 数组）
        const IndexT eE0 = (IndexT)effExtents[0];
        const IndexT eE1 = (IndexT)effExtents[1];
        const IndexT eE2 = (IndexT)effExtents[2];
        const IndexT eE3 = (IndexT)effExtents[3];
        const IndexT eS0 = (IndexT)effStrides[0];
        const IndexT eS1 = (IndexT)effStrides[1];
        const IndexT eS2 = (IndexT)effStrides[2];
        const IndexT eS3 = (IndexT)effStrides[3];
        const IndexT eS4 = (IndexT)effStrides[4];

        uint32_t remain = sizeof(T) > 4 ? totalGroups * 2 : totalGroups;

        // ============ Gather→Scatter: left/mid/right 三阶段合一，baseIdx 只算一次 ============
        __VEC_SCOPE__
        {
            Reg::MaskReg mask;
            Reg::RegTensor<RangeT> tmpRange;
            Reg::RegTensor<IndexT> baseIdx, quot, idx, tmpReg;
            Reg::RegTensor<IndexT> qNext, ik, dExt;
            Reg::RegTensor<T> vregT, vregPadT;
            Reg::RegTensor<T> vregOut, vregScatter; // B8 Pack/UnPack 中转
            for (uint16_t b = 0; b < numBatches; b++) {
                mask = Reg::UpdateMask<IndexT>(remain);
                // === shared baseIdx computation ===
                Reg::Arange(tmpRange, (RangeT)0);
                quot = (Reg::RegTensor<IndexT>&)tmpRange;
                Reg::Adds(quot, quot, (IndexT)((RangeT)b * (RangeT)kBatchVL), mask);
                Reg::Duplicate(baseIdx, (IndexT)baseE, mask);

                if constexpr (kEffAxes >= 2) {
                    Reg::Duplicate(dExt, eE0, mask);
                    Reg::Div<IndexT>(qNext, quot, dExt, mask);
                    Reg::Muls(tmpReg, qNext, eE0, mask);
                    Reg::Sub<IndexT>(ik, quot, tmpReg, mask);
                    Reg::Muls(tmpReg, ik, eS0, mask);
                    Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                    Reg::Copy(quot, qNext);
                    if constexpr (kEffAxes >= 3) {
                        Reg::Duplicate(dExt, eE1, mask);
                        Reg::Div<IndexT>(qNext, quot, dExt, mask);
                        Reg::Muls(tmpReg, qNext, eE1, mask);
                        Reg::Sub<IndexT>(ik, quot, tmpReg, mask);
                        Reg::Muls(tmpReg, ik, eS1, mask);
                        Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                        Reg::Copy(quot, qNext);
                        if constexpr (kEffAxes >= 4) {
                            Reg::Duplicate(dExt, eE2, mask);
                            Reg::Div<IndexT>(qNext, quot, dExt, mask);
                            Reg::Muls(tmpReg, qNext, eE2, mask);
                            Reg::Sub<IndexT>(ik, quot, tmpReg, mask);
                            Reg::Muls(tmpReg, ik, eS2, mask);
                            Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                            Reg::Copy(quot, qNext);
                            if constexpr (kEffAxes >= 5) {
                                Reg::Duplicate(dExt, eE3, mask);
                                Reg::Div<IndexT>(qNext, quot, dExt, mask);
                                Reg::Muls(tmpReg, qNext, eE3, mask);
                                Reg::Sub<IndexT>(ik, quot, tmpReg, mask);
                                Reg::Muls(tmpReg, ik, eS3, mask);
                                Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                                Reg::Copy(quot, qNext);
                                Reg::Muls(tmpReg, quot, eS4, mask);
                                Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                            } else {
                                Reg::Muls(tmpReg, quot, eS3, mask);
                                Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                            }
                        } else {
                            Reg::Muls(tmpReg, quot, eS2, mask);
                            Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                        }
                    } else {
                        Reg::Muls(tmpReg, quot, eS1, mask);
                        Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                    }
                } else {
                    Reg::Muls(tmpReg, quot, eS0, mask);
                    Reg::Add<IndexT>(baseIdx, baseIdx, tmpReg, mask);
                }

                // ======== Phase A: 左边界（iN1=0） ========
                Reg::Adds(idx, baseIdx, (IndexT)pL, mask);
                if constexpr (sizeof(T) == 1) {
                    Reg::DataCopyGather((Reg::RegTensor<CastT>&)vregT, dataAddr, idx, mask);
                } else {
                    Reg::DataCopyGather(vregT, dataAddr, idx, mask);
                }

                for (uint16_t k = 0; k < phaseA_pLu; k++) {
                    Reg::Adds(idx, baseIdx, (IndexT)k, mask);
                    if constexpr (sizeof(T) == 1) {
                        Reg::DataCopyGather((Reg::RegTensor<CastT>&)vregPadT, dataAddr, idx, mask);
                        Reg::Add((Reg::RegTensor<CastT>&)vregT, (Reg::RegTensor<CastT>&)vregT,
                                 (Reg::RegTensor<CastT>&)vregPadT, mask);
                    } else {
                        Reg::DataCopyGather(vregPadT, dataAddr, idx, mask);
                        Reg::Add(vregT, vregT, vregPadT, mask);
                    }
                }

                Reg::Arange(tmpRange, (RangeT)0);
                Reg::Muls(idx, (Reg::RegTensor<IndexT>&)tmpRange, (IndexT)outRowE, mask);
                Reg::Adds(idx, idx, (IndexT)((RangeT)b * (RangeT)kBatchVL * outRowE), mask);

                if constexpr (sizeof(T) == 1) {
                    Reg::Pack(vregOut, (Reg::RegTensor<CastT>&)vregT);
                    Reg::UnPack((Reg::RegTensor<CastT>&)vregScatter, vregOut);
                    Reg::DataCopyScatter(outputAddr, vregScatter, idx, mask);
                } else {
                    Reg::DataCopyScatter(outputAddr, vregT, idx, mask);
                }

                // ======== Phase B: 中间（iN1 ∈ [1, dN1-1)） ========
                for (uint16_t i = 0; i < midCount; i++) {
                    const uint16_t iN1 = (uint16_t)(i + 1);
                    Reg::Adds(idx, baseIdx, (IndexT)((RangeT)iN1 + (RangeT)pL), mask);
                    if constexpr (sizeof(T) == 1) {
                        Reg::DataCopyGather((Reg::RegTensor<CastT>&)vregT, dataAddr, idx, mask);
                    } else {
                        Reg::DataCopyGather(vregT, dataAddr, idx, mask);
                    }

                    Reg::Arange(tmpRange, (RangeT)0);
                    Reg::Muls(idx, (Reg::RegTensor<IndexT>&)tmpRange, (IndexT)outRowE, mask);
                    Reg::Adds(idx, idx, (IndexT)((RangeT)b * (RangeT)kBatchVL * outRowE + (RangeT)iN1), mask);
                    if constexpr (sizeof(T) == 1) {
                        Reg::Pack(vregOut, (Reg::RegTensor<CastT>&)vregT);
                        Reg::UnPack((Reg::RegTensor<CastT>&)vregScatter, vregOut);
                        Reg::DataCopyScatter(outputAddr, vregScatter, idx, mask);
                    } else {
                        Reg::DataCopyScatter(outputAddr, vregT, idx, mask);
                    }
                }

                // ======== Phase C: 右边界（iN1 = dN1-1） ========
                Reg::Adds(idx, baseIdx, (IndexT)((RangeT)pL + (RangeT)iRight), mask);
                if constexpr (sizeof(T) == 1) {
                    Reg::DataCopyGather((Reg::RegTensor<CastT>&)vregT, dataAddr, idx, mask);
                } else {
                    Reg::DataCopyGather(vregT, dataAddr, idx, mask);
                }

                // dN1==1: re-accumulate left padding (otherwise lost by Phase C overwrite)
                for (uint16_t k = 0; k < combPadCnt; k++) {
                    Reg::Adds(idx, baseIdx, (IndexT)k, mask);
                    if constexpr (sizeof(T) == 1) {
                        Reg::DataCopyGather((Reg::RegTensor<CastT>&)vregPadT, dataAddr, idx, mask);
                        Reg::Add((Reg::RegTensor<CastT>&)vregT, (Reg::RegTensor<CastT>&)vregT,
                                 (Reg::RegTensor<CastT>&)vregPadT, mask);
                    } else {
                        Reg::DataCopyGather(vregPadT, dataAddr, idx, mask);
                        Reg::Add(vregT, vregT, vregPadT, mask);
                    }
                }

                for (uint16_t k = 0; k < pRu; k++) {
                    Reg::Adds(idx, baseIdx, (IndexT)(rightPadStart + (RangeT)k), mask);
                    if constexpr (sizeof(T) == 1) {
                        Reg::DataCopyGather((Reg::RegTensor<CastT>&)vregPadT, dataAddr, idx, mask);
                        Reg::Add((Reg::RegTensor<CastT>&)vregT, (Reg::RegTensor<CastT>&)vregT,
                                 (Reg::RegTensor<CastT>&)vregPadT, mask);
                    } else {
                        Reg::DataCopyGather(vregPadT, dataAddr, idx, mask);
                        Reg::Add(vregT, vregT, vregPadT, mask);
                    }
                }

                Reg::Arange(tmpRange, (RangeT)0);
                Reg::Muls(idx, (Reg::RegTensor<IndexT>&)tmpRange, (IndexT)outRowE, mask);
                Reg::Adds(idx, idx, (IndexT)((RangeT)b * (RangeT)kBatchVL * outRowE + (RangeT)iRight), mask);
                if constexpr (sizeof(T) == 1) {
                    Reg::Pack(vregOut, (Reg::RegTensor<CastT>&)vregT);
                    Reg::UnPack((Reg::RegTensor<CastT>&)vregScatter, vregOut);
                    Reg::DataCopyScatter(outputAddr, vregScatter, idx, mask);
                } else {
                    Reg::DataCopyScatter(outputAddr, vregT, idx, mask);
                }
            }
        }
    }

    // 计算 dataBuf 内多维位置（origCoords）的元素偏移。
    // dataBuf 布局：外层轴 0..SPLIT_AXIS-1 按 extent 紧密排布，SPLIT_AXIS / 中间维 / 尾轴按
    // strideAligned_ 排布（尾轴 32B 对齐）。axis k 的 stride 走 SliceStrideInBuf。
    // rel[k] = (origCoords[k] + leftPad[k] - loadStart[k])：把原始（不含 pad）坐标转换为
    //          dataBuf 中 padding 后的相对坐标。
    __aicore__ inline uint64_t ComputeRowOffsetInBuf(const uint32_t* origCoords, const uint32_t* loadStart,
                                                     const uint32_t* loadEnd) const
    {
        uint64_t ubOff = 0;
        for (int k = (int)DIM_NUM - 2; k >= 0; k--) {
            int32_t j_k = (int32_t)origCoords[k] + leftPad_[k];
            uint32_t rel = (uint32_t)(j_k - (int32_t)loadStart[k]);
            ubOff += (uint64_t)rel * SliceStrideInBuf((uint32_t)k, loadStart, loadEnd);
        }
        return ubOff;
    }

    // ---------------- outputBuf → grad_in GM ----------------
    __aicore__ inline void OutputTileToGm(const uint32_t* outerCoords, uint32_t tileStart, uint32_t tileLen)
    {
        // grad_in GM offset（按 inputShape 累乘）
        uint64_t gmOffset = 0;
        uint64_t stride = 1;
        for (int k = (int)DIM_NUM - 1; k >= 0; k--) {
            uint32_t coord = 0;
            if (k == (int)SPLIT_AXIS)
                coord = tileStart;
            else if (k < (int)SPLIT_AXIS)
                coord = outerCoords[k];
            gmOffset += (uint64_t)coord * stride;
            stride *= inputShape_[k];
        }

        uint64_t outElems = (uint64_t)tileLen * innerProdInGm_;
        uint64_t outBytes = outElems * sizeof(T);

        LocalTensor<T> outLocal = outputBuf_.Get<T>();
        DataCopyExtParams params{1, (uint32_t)outBytes, 0, 0, 0};
        DataCopyPad(gradInGm_[gmOffset], outLocal, params);
    }

private:
    TPipe* pipe_;
    const PadV3GradReplicationTilingData* tilingData_;

    GlobalTensor<T> gradOutGm_;
    GlobalTensor<T> gradInGm_;

    TBuf<TPosition::VECCALC> dataBuf_;   // kNeedCast 时按 PromoteT 分配，否则按 T
    TBuf<TPosition::VECCALC> outputBuf_; // T 大小

    uint32_t blockIdx_;
    uint64_t inputShape_[MAX_DIMS] = {0};
    uint64_t outputShape_[MAX_DIMS] = {0};
    uint64_t strideAligned_[MAX_DIMS] = {0};
    int32_t leftPad_[MAX_DIMS] = {0};
    int32_t rightPad_[MAX_DIMS] = {0};

    uint32_t splitSize_ = 0;
    uint64_t splitCount_ = 0;
    uint32_t tilesPerCore_ = 0;
    uint64_t splitCountAxis_ = 0;
    uint64_t worstFactor_ = 1;
    // UB 内 axis>SPLIT_AXIS 总 element 数（尾轴 32B 对齐）。SPLIT_AXIS<N-1 时 = strideAligned_[SPLIT_AXIS]。
    uint64_t innerProdUb_ = 1;
    // GM 紧凑 axis>SPLIT_AXIS 总 element 数（=∏ outputShape[k>SPLIT_AXIS]）。
    // grad_in GM 紧凑 axis>SPLIT_AXIS 总 element 数（=∏ inputShape[k>SPLIT_AXIS]）。
    uint64_t innerProdInGm_ = 1;

    uint64_t dataBufElems_ = 0; // dataBuf T 元素数（kNeedCast 时用于 BulkCast 长度）
    uint64_t dataBufBytes_ = 0;
    uint64_t outputBufBytes_ = 0;
};

} // namespace PadV3GradReplication

#endif // PAD_V3_GRAD_REPLICATION_H
