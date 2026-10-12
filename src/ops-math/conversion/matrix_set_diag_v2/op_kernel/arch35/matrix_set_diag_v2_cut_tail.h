/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_SET_DIAG_CUT_TAIL_H
#define MATRIX_SET_DIAG_CUT_TAIL_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "matrix_set_diag_v2_tilingdata.h"
#include "matrix_set_diag_v2_simt.h"

namespace MSD {
using namespace AscendC;

/**
 * x 切分的形状
 */
enum class XTilingShape {
    HorizRect, // 水平矩形，col > row
    VertRect,  // 垂直矩形，col <= row
};

/**
 * 对角线在x上的扩展的方向
 */
enum class ExpandDirection {
    Horiz, // 横向
    Vert,  // 纵向
};

struct MSDCutTailBlockInfo {
    uint64_t batch{0};
    uint64_t startRow{0};
    uint64_t endRow{0};
    uint64_t startCol{0};
    uint64_t endCol{0};
    uint32_t rowCnt{0};
    uint32_t colCnt{0};
    int64_t upperLeftK{0};
    int64_t upperRightK{0};
    int64_t bottomLeftK{0};
    int64_t bottomRightK{0};
    int32_t startK{0};
    int32_t endK{0};
    uint32_t diagLen{0};
    uint32_t diagCnt{0};
    uint32_t diagSize{0};
    int32_t rowOffset{0};
    MutexID eventID{0};
};

template <typename T, typename U>
class MatrixSetDiagCutTail {
private:
    // 类型声明
    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using MaskType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

private:
    constexpr static uint32_t D_SIZE = sizeof(T);
    constexpr static Reg::LoadDist LOAD_DIST = D_SIZE == 1 ? Reg::LoadDist::DIST_UNPACK_B8 : Reg::LoadDist::DIST_NORM;
    constexpr static Reg::StoreDist STORE_DIST = D_SIZE == 1 ? Reg::StoreDist::DIST_PACK_B16 :
                                                               Reg::StoreDist::DIST_NORM;
    constexpr static Reg::RegTrait REG_TRAIT = D_SIZE == sizeof(int64_t) ? Reg::RegTraitNumTwo : Reg::RegTraitNumOne;
    constexpr static uint32_t BLOCK_ALIGN_ELEMENTS = Ops::Base::GetUbBlockSize() / D_SIZE; // block 对齐元素数
    constexpr static uint32_t BUF_NUM = 2;                                                 // buffer 数量
    constexpr static uint32_t VREG_ELEMENTS = Ops::Base::GetVRegSize() / sizeof(IdxType);  // vreg 元素数量
    constexpr static uint32_t SIMD_MIN_DIAG_SIZE = VREG_ELEMENTS; // 走 simd 的最小diag大小
    constexpr static uint32_t SIMD_MIN_DIAG_LEN = 8 / D_SIZE;     // 走 simd 的最小diag长度
    constexpr static uint32_t GATHER_SCATTER_MAX_DSIZE = 2;       // 走 gather+scatter 的最大dtype size
    constexpr static uint32_t X_MAX_BUF_SIZE = 32 * 1024;
    constexpr static uint32_t X_MAX_ELEMENTS = X_MAX_BUF_SIZE / D_SIZE;
    constexpr static uint32_t DIAG_MAX_BUF_SIZE = 64 * 1024;
    constexpr static uint32_t DIAG_MAX_ELEMENTS = DIAG_MAX_BUF_SIZE / D_SIZE;
    constexpr static uint32_t BANK_OFFSET = 256;
    constexpr static uint32_t BANK_CONFLIC_STRIDE = 512;
    constexpr static uint32_t BANK_CONFLIC_STRIDE_ELE_NUM = BANK_CONFLIC_STRIDE / D_SIZE;
    constexpr static uint32_t MAX_DISCRETE_BANK_CFLT_CNT = 4;

    template <uint32_t N>
    using SimtSetDiagByRowFunc = SimtCutTailSetDiagByRowFunc<T, U, N>;
    template <uint32_t N>
    using SimtSetDiagByColFunc = SimtCutTailSetDiagByColFunc<T, U, N>;

private:
    // gm 地址
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> diagGm_;
    GlobalTensor<T> outputGm_;

    // 队列
    TBuf<TPosition::VECCALC> buf_;

    // 输入数据
    TPipe* pipe_{nullptr};
    uint32_t blockIdx_{0};
    uint32_t coreNum_{0};    // 核数
    uint64_t batchSize_{0};  // batch 维度的大小
    uint64_t xRowNum_{0};    // x的行数
    uint64_t xColNum_{0};    // x的列数
    int32_t k0_{0};          // 下对角线偏移
    int32_t k1_{0};          // 上对角线偏移
    uint32_t maxDiagLen_{0}; // 最大对角线长度
    uint64_t ubPerCore_{0};  // 单核处理的ub个数
    uint32_t xRowFactor_{0}; // 一次处理的行数
    uint32_t xColFactor_{0}; // 一次处理的列数
    uint64_t diagRowMin_{0}; // 含对角线的起始行
    uint64_t diagRowMax_{0}; // 含对角线的终止行
    uint64_t diagColMin_{0}; // 含对角线的起始列
    uint64_t diagColMax_{0}; // 含对角线的终止列

    // 中间变量
    uint32_t mainBlockElements_{0};
    uint64_t diagElementsPerBatch_{0};
    bool scatterBankConflict_{false};

    // 搬运参数
    const DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};

public:
    __aicore__ inline MatrixSetDiagCutTail(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, const MSDV2CutTailTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <XTilingShape xTilingShape>
    __aicore__ inline void ProcessWithXShape();
    // 搬运
    __aicore__ inline void CopyInX(LocalTensor<T>& xTensor, uint64_t xStart, uint32_t xNum);
    __aicore__ inline void CopyInX(LocalTensor<T>& xTensor, uint64_t xStart, uint32_t xNum, MSDCutTailBlockInfo& info);
    __aicore__ inline void CopyInDiagHoriz(LocalTensor<T>& diagTensor, uint64_t diagStart, MSDCutTailBlockInfo& info,
                                           uint32_t& diagIdxOffset, uint32_t& ubAglinStart, uint32_t& ubAglinOffset);
    __aicore__ inline void CopyInDiagVert(LocalTensor<T>& diagTensor, uint64_t diagStart, MSDCutTailBlockInfo& info,
                                          uint32_t& diagIdxOffset, uint32_t& ubAglinStart, uint32_t& ubAglinOffset);
    template <ExpandDirection direction>
    __aicore__ inline void CopyInDiag(LocalTensor<T>& diagTensor, uint64_t diagStart, MSDCutTailBlockInfo& info,
                                      uint32_t& diagIdxOffset, uint32_t& ubAglinStart, uint32_t& ubAglinOffset);
    __aicore__ inline void CopyOut(LocalTensor<T>& diagTensor, uint64_t yStart, uint32_t yNum);
    __aicore__ inline void OptimizeCopyParams(DataCopyExtParams& params);
    // 参数计算
    __aicore__ inline int64_t ComputeUpperLeftK(const MSDCutTailBlockInfo& info) const;
    __aicore__ inline int64_t ComputeUpperRightK(const MSDCutTailBlockInfo& info) const;
    __aicore__ inline int64_t ComputeBottomLeftK(const MSDCutTailBlockInfo& info) const;
    __aicore__ inline int64_t ComputeBottomRightK(const MSDCutTailBlockInfo& info) const;
    __aicore__ inline bool IsServeConflict(uint32_t stride, uint32_t count) const;
    template <ExpandDirection direction>
    __aicore__ inline bool IsGatherServeConflict(const MSDCutTailBlockInfo& info) const;
    __aicore__ inline bool IsScatterServeConflict(const MSDCutTailBlockInfo& info) const;
    template <XTilingShape xTilingShape>
    __aicore__ inline void Compute(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor, MSDCutTailBlockInfo& info);
    template <ExpandDirection direction>
    __aicore__ inline bool ShouldUseSimt(const MSDCutTailBlockInfo& info) const;
    template <ExpandDirection direction>
    __aicore__ inline bool ShouldUseGather(const MSDCutTailBlockInfo& info) const;
    template <ExpandDirection direction>
    __aicore__ inline bool ShouldUseGatherScatter(const MSDCutTailBlockInfo& info) const;
    template <ExpandDirection direction>
    __aicore__ inline bool ShouldUseScatter(const MSDCutTailBlockInfo& info) const;
    template <ExpandDirection direction>
    __aicore__ inline void CopyInDiagAndCompute(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                                MSDCutTailBlockInfo& info);
    __aicore__ inline void ComputeBySimt(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                         MSDCutTailBlockInfo& info);
    template <ExpandDirection direction>
    __aicore__ inline void ComputeByScatter(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                            MSDCutTailBlockInfo& info);
    template <ExpandDirection direction>
    __aicore__ inline void ComputeByGather(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                           MSDCutTailBlockInfo& info);
    template <ExpandDirection direction>
    __aicore__ inline void ComputeByGatherScatter(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                                  MSDCutTailBlockInfo& info);
    // simd vf
    static __simd_callee__ inline void SimdDoScatter(__ubuf__ T* x, Reg::RegTensor<T, REG_TRAIT>& diagReg,
                                                     Reg::RegTensor<IdxType>& xRow, Reg::RegTensor<IdxType>& xCol,
                                                     Reg::RegTensor<IdxType>& colCnt, Reg::MaskReg& validMask);
    template <ExpandDirection direction>
    static __simd_vf__ inline void SimdSetDiagScatter(__ubuf__ T* x, __ubuf__ T* diag, uint32_t diagLen,
                                                      uint16_t loopNum, uint32_t processNum, uint32_t rowCnt,
                                                      uint32_t colCnt, RangeType rowOffset, IdxType diagIdxOffset,
                                                      IdxType ubAglinStart, IdxType ubAglinOffset);
    template <ExpandDirection direction>
    static __simd_vf__ inline void SimdSetDiagGather(__ubuf__ T* x, __ubuf__ T* diag, uint32_t diagLen,
                                                     uint16_t loopNum, uint32_t processNum, uint32_t rowCnt,
                                                     uint32_t colCnt, RangeType rowOffset, IdxType diagCnt,
                                                     IdxType diagIdxOffset, IdxType ubAglinStart,
                                                     IdxType ubAglinOffset);
    template <ExpandDirection direction>
    static __simd_vf__ inline void SimdSetDiagGatherScatter(__ubuf__ T* x, __ubuf__ T* diag, uint32_t diagLen,
                                                            uint16_t loopNum, uint32_t processNum, uint32_t rowCnt,
                                                            uint32_t colCnt, uint32_t diagCnt, RangeType colOffset,
                                                            IdxType diagCntIdx, IdxType diagIdxOffset,
                                                            IdxType ubAglinStart, IdxType ubAglinOffset);
};

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::Init(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y,
                                                        const MSDV2CutTailTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    // gm
    inputGm_.SetGlobalBuffer((__gm__ T*)x);
    diagGm_.SetGlobalBuffer((__gm__ T*)diagonal);
    outputGm_.SetGlobalBuffer((__gm__ T*)y);

    // tiling
    coreNum_ = tilingData->input.coreNum;
    batchSize_ = tilingData->input.mergeDimSize;
    xRowNum_ = tilingData->input.xRowNum;
    xColNum_ = tilingData->input.xColNum;
    k0_ = tilingData->input.k0;
    k1_ = tilingData->input.k1;
    maxDiagLen_ = tilingData->input.maxDiagLen;
    ubPerCore_ = tilingData->totalCntPerCore;
    xRowFactor_ = tilingData->xRowFactor;
    xColFactor_ = tilingData->xColFactor;
    diagRowMin_ = Std::max(0, -k1_);
    diagRowMax_ = Std::min(xRowNum_ - 1, xColNum_ - 1 - k0_);
    diagColMin_ = Std::max(0, k0_);
    diagColMax_ = Std::min(xColNum_ - 1, xRowNum_ - 1 + k1_);

    // 中间变量
    diagElementsPerBatch_ = maxDiagLen_ * static_cast<uint64_t>(static_cast<int64_t>(k1_) - k0_ + 1);
    mainBlockElements_ = xRowFactor_ * xColFactor_;
    scatterBankConflict_ = (xColNum_ + 1) % BANK_CONFLIC_STRIDE_ELE_NUM == 0;

    // ub
    pipe_->InitBuffer(buf_, BUF_NUM * (X_MAX_BUF_SIZE + DIAG_MAX_BUF_SIZE + BANK_OFFSET));
}

template <typename T, typename U>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::IsServeConflict(uint32_t stride, uint32_t count) const
{
    // 512倍数，则有冲突
    uint32_t v = stride & (BANK_CONFLIC_STRIDE - 1);
    if (v == 0) {
        return true;
    }

    // 计算周期长度
    uint32_t k = 0;
    if ((v & 0xFF) == 0) {
        k += 8;
        v >>= 8;
    }
    if ((v & 0x0F) == 0) {
        k += 4;
        v >>= 4;
    }
    if ((v & 0x03) == 0) {
        k += 2;
        v >>= 2;
    }
    if ((v & 0x01) == 0) {
        k += 1;
    }
    uint32_t tLen = 1U << (9 - k);

    // 超过最大数量即严重冲突
    return Ops::Base::CeilDiv(count, tLen) > MAX_DISCRETE_BANK_CFLT_CNT;
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::IsGatherServeConflict(const MSDCutTailBlockInfo& info) const
{
    if constexpr (direction == ExpandDirection::Horiz) {
        return IsServeConflict(info.diagLen * D_SIZE, info.diagCnt);
    } else {
        return IsServeConflict((info.diagLen - 1) * D_SIZE, info.diagCnt);
    }
}

template <typename T, typename U>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::IsScatterServeConflict(const MSDCutTailBlockInfo& info) const
{
    return IsServeConflict((xColNum_ + 1) * D_SIZE, info.diagLen);
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::OptimizeCopyParams(DataCopyExtParams& params)
{
    // stride 为0，则直接连续搬即可
    if (params.srcStride == 0) {
        params.blockLen *= params.blockCount;
        params.blockCount = 1;
    }
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyInX(LocalTensor<T>& xTensor, uint64_t xStart, uint32_t xNum)
{
    DataCopyExtParams xCopyInParams{1u, static_cast<uint32_t>(xNum * D_SIZE), 0, 0, 0};
    DataCopyPad(xTensor, inputGm_[xStart], xCopyInParams, padParams_);
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyInX(LocalTensor<T>& xTensor, uint64_t xStart, uint32_t xNum,
                                                           MSDCutTailBlockInfo& info)
{
    if (info.endK == info.upperRightK) {
        if (info.startK == info.bottomLeftK) {
            // 全是对角线就不用搬了
            return;
        }
        if (info.upperLeftK >= info.startK) {
            // 前面的部分被对角线覆盖，无需搬入
            uint32_t offset = Ops::Base::FloorAlign(
                static_cast<uint32_t>(info.upperLeftK - info.startK + 1) * info.colCnt, BLOCK_ALIGN_ELEMENTS);
            LocalTensor<T> xPart = xTensor[offset];
            CopyInX(xPart, xStart + offset, xNum - offset);
            return;
        }
    }
    CopyInX(xTensor, xStart, xNum);
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyInDiagHoriz(LocalTensor<T>& diagTensor, uint64_t diagStart,
                                                                   MSDCutTailBlockInfo& info, uint32_t& diagIdxOffset,
                                                                   uint32_t& ubAglinStart, uint32_t& ubAglinOffset)
{
    uint32_t stride = static_cast<uint32_t>(maxDiagLen_ - info.diagLen) * D_SIZE;
    uint32_t blockLen = info.diagLen * D_SIZE;
    if (info.startK >= 0) {
        // 上对角线，也有可能是右边界或上边界的一小部分
        diagStart += info.startRow;
        DataCopyExtParams topDiagCopyInParams{static_cast<uint16_t>(info.diagCnt), blockLen, stride, 0, 0};
        OptimizeCopyParams(topDiagCopyInParams);
        DataCopyPad<T, PaddingMode::Compact>(diagTensor, diagGm_[diagStart], topDiagCopyInParams, padParams_);
        return;
    }
    if (info.endK < 0) {
        // 下对角线
        diagStart += info.startRow + info.endK;
        DataCopyExtParams bottomDiagCopyInParams{static_cast<uint16_t>(info.diagCnt), blockLen, stride - D_SIZE, 0, 0};
        OptimizeCopyParams(bottomDiagCopyInParams);
        DataCopyPad<T, PaddingMode::Compact>(diagTensor, diagGm_[diagStart], bottomDiagCopyInParams, padParams_);
        return;
    }
    // 上对角线
    diagStart += info.startRow;
    DataCopyExtParams topDiagCopyInParams{static_cast<uint16_t>(info.endK + 1), blockLen, stride, 0, 0};
    DataCopyPad<T, PaddingMode::Compact>(diagTensor, diagGm_[diagStart], topDiagCopyInParams, padParams_);
    // 下对角线
    diagStart += (info.endK + 1) * maxDiagLen_ - 1;
    DataCopyExtParams bottomDiagCopyInParams{static_cast<uint16_t>(-info.startK), blockLen, stride - D_SIZE, 0, 0};
    OptimizeCopyParams(bottomDiagCopyInParams);
    ubAglinStart = topDiagCopyInParams.blockCount * info.diagLen;
    uint32_t upDiagSizeAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(ubAglinStart), BLOCK_ALIGN_ELEMENTS);
    ubAglinOffset = upDiagSizeAlign - ubAglinStart;
    DataCopyPad<T, PaddingMode::Compact>(diagTensor[upDiagSizeAlign], diagGm_[diagStart], bottomDiagCopyInParams,
                                         padParams_);
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyInDiagVert(LocalTensor<T>& diagTensor, uint64_t diagStart,
                                                                  MSDCutTailBlockInfo& info, uint32_t& diagIdxOffset,
                                                                  uint32_t& ubAglinStart, uint32_t& ubAglinOffset)
{
    if (info.startK > 0) {
        // 上对角线，不包括主对角线
        diagIdxOffset = info.endK;
        DataCopyExtParams topDiagCopyInParams{static_cast<uint16_t>(info.diagCnt), (info.colCnt - 1) * D_SIZE, D_SIZE,
                                              0, 0};
        DataCopyPad<T, PaddingMode::Compact>(diagTensor, diagGm_[diagStart], topDiagCopyInParams, padParams_);
        return;
    }
    if (info.endK <= 0) {
        // 下对角线 + 主对角线，也有可能是左边界或下边界的一小部分
        DataCopyExtParams bottomDiagCopyInParams{static_cast<uint16_t>(info.diagCnt), info.diagLen * D_SIZE,
                                                 static_cast<uint32_t>(maxDiagLen_ - info.diagLen) * D_SIZE, 0, 0};
        OptimizeCopyParams(bottomDiagCopyInParams);
        DataCopyPad<T, PaddingMode::Compact>(diagTensor, diagGm_[diagStart], bottomDiagCopyInParams, padParams_);
        return;
    }
    // 上对角线，不包括主对角线
    diagIdxOffset = info.endK;
    DataCopyExtParams topDiagCopyInParams{static_cast<uint16_t>(info.endK), (info.colCnt - 1) * D_SIZE, D_SIZE, 0, 0};
    DataCopyPad<T, PaddingMode::Compact>(diagTensor, diagGm_[diagStart], topDiagCopyInParams, padParams_);
    // 下对角线 + 主对角线，直接一次性搬
    diagStart += info.endK * maxDiagLen_;
    DataCopyExtParams bottomDiagCopyInParams{1, (1 - info.startK) * info.colCnt * D_SIZE, 0, 0, 0};
    ubAglinStart = topDiagCopyInParams.blockCount * (info.colCnt - 1);
    uint32_t upDiagSizeAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(ubAglinStart), BLOCK_ALIGN_ELEMENTS);
    ubAglinOffset = upDiagSizeAlign - ubAglinStart;
    DataCopyPad<T, PaddingMode::Compact>(diagTensor[upDiagSizeAlign], diagGm_[diagStart], bottomDiagCopyInParams,
                                         padParams_);
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyInDiag(LocalTensor<T>& diagTensor, uint64_t diagStart,
                                                              MSDCutTailBlockInfo& info, uint32_t& diagIdxOffset,
                                                              uint32_t& ubAglinStart, uint32_t& ubAglinOffset)
{
    if constexpr (direction == ExpandDirection::Horiz) {
        CopyInDiagHoriz(diagTensor, diagStart, info, diagIdxOffset, ubAglinStart, ubAglinOffset);
    } else {
        CopyInDiagVert(diagTensor, diagStart, info, diagIdxOffset, ubAglinStart, ubAglinOffset);
    }
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyOut(LocalTensor<T>& yTensor, uint64_t yStart, uint32_t yNum)
{
    DataCopyExtParams yCopyOutParams{1u, static_cast<uint32_t>(yNum * D_SIZE), 0, 0, 0};
    DataCopyPad(outputGm_[yStart], yTensor, yCopyOutParams);
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::ComputeBySimt(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                                                 MSDCutTailBlockInfo& info)
{
    U startRow = Std::max(info.startRow, diagRowMin_);
    U endRow = Std::min(info.endRow, diagRowMax_);
    U startCol = Std::max(info.startCol, diagColMin_);
    U endCol = Std::min(info.endCol, diagColMax_);
    if (startRow > endRow || startCol > endCol) {
        Mutex::Unlock<PIPE_MTE2>(info.eventID);
        Mutex::Lock<PIPE_V>(info.eventID);
        return;
    }

    uint64_t diagStart = info.batch * diagElementsPerBatch_ + static_cast<uint64_t>(k1_ - info.endK) * maxDiagLen_;
    Mutex::Unlock<PIPE_MTE2>(info.eventID);
    Mutex::Lock<PIPE_V>(info.eventID);

    U rowLen = endRow - startRow;
    U colLen = endCol - startCol;
    U kLen = info.diagCnt;
    int32_t endK = info.endK;
    uint32_t magic = 0, shift = 0;
    GetUintDivMagicAndShift(magic, shift, static_cast<uint32_t>(kLen));
    __ubuf__ T* xPtr = (__ubuf__ T*)(xTensor.GetPhyAddr());

    U workSize = (rowLen <= colLen) ? kLen * (endRow - startRow + 1) : kLen * (endCol - startCol + 1);

    __gm__ T* diagGmPtr = (__gm__ T*)(diagGm_[diagStart].GetPhyAddr());
    if (rowLen <= colLen) {
        SimtDispatch<SimtSetDiagByRowFunc>(static_cast<uint32_t>(workSize), (U)startRow, (U)endRow, (U)startCol,
                                           (U)endCol, (U)info.startRow, (U)info.startCol, endK, kLen, magic, shift,
                                           diagGmPtr, xPtr, xColNum_, maxDiagLen_);
    } else {
        SimtDispatch<SimtSetDiagByColFunc>(static_cast<uint32_t>(workSize), (U)startRow, (U)endRow, (U)startCol,
                                           (U)endCol, (U)info.startRow, (U)info.startCol, endK, kLen, magic, shift,
                                           diagGmPtr, xPtr, xColNum_, maxDiagLen_);
    }
}

template <typename T, typename U>
__simd_callee__ inline void MatrixSetDiagCutTail<T, U>::SimdDoScatter(
    __ubuf__ T* x, Reg::RegTensor<T, REG_TRAIT>& diagReg, Reg::RegTensor<IdxType>& xRow, Reg::RegTensor<IdxType>& xCol,
    Reg::RegTensor<IdxType>& colCnt, Reg::MaskReg& validMask)
{
    // 计算 x_idx = x_row * col_len + x_col
    Reg::RegTensor<IdxType>& xIdx = xCol;
    Reg::MulAddDst(xIdx, xRow, colCnt, validMask);

    Reg::Scatter(x, diagReg, xIdx, validMask);
}

template <typename T, typename U>
template <ExpandDirection direction>
__simd_vf__ inline void MatrixSetDiagCutTail<T, U>::SimdSetDiagScatter(
    __ubuf__ T* x, __ubuf__ T* diag, uint32_t diagLen, uint16_t loopNum, uint32_t processNum, uint32_t rowCnt,
    uint32_t colCnt, RangeType rowOffset, IdxType diagIdxOffset, IdxType ubAglinStart, IdxType ubAglinOffset)
{
    Reg::RegTensor<T, REG_TRAIT> diagReg;
    Reg::RegTensor<IdxType> diagIdx;
    Reg::RegTensor<IdxType> diagRow;
    Reg::RegTensor<IdxType> diagCol;
    Reg::RegTensor<IdxType> diagLenReg;
    Reg::RegTensor<IdxType> ubAglinOffsetReg;
    Reg::RegTensor<IdxType> colCntReg;
    Reg::RegTensor<IdxType> tmp;
    Reg::MaskReg mask;
    Reg::MaskReg bottomDiagMask;
    Reg::MaskReg topDiagMask;
    Reg::MaskReg validMask;

    Reg::Duplicate(diagLenReg, (IdxType)diagLen);
    Reg::Duplicate(colCntReg, (IdxType)colCnt);
    for (uint16_t i = 0; i < loopNum; i++) {
        mask = Reg::UpdateMask<MaskType>(processNum);
        Reg::LoadAlign<T, LOAD_DIST>(diagReg, diag + i * VREG_ELEMENTS);
        Reg::Arange((Reg::RegTensor<RangeType>&)diagIdx, i * VREG_ELEMENTS);

        // 排除对齐的部分
        Reg::Compares<IdxType, CMPMODE::GE>(bottomDiagMask, diagIdx, ubAglinStart + ubAglinOffset, mask);
        Reg::Compares<IdxType, CMPMODE::LT>(topDiagMask, diagIdx, ubAglinStart, mask);
        Reg::Or(validMask, topDiagMask, bottomDiagMask, mask);

        // 计算实际的索引
        // 下对角线要减去对齐的偏移
        Reg::Duplicate(ubAglinOffsetReg, -ubAglinOffset, bottomDiagMask);
        Reg::Add(diagIdx, diagIdx, ubAglinOffsetReg, mask);
        // 起始偏移
        Reg::Adds(diagIdx, diagIdx, diagIdxOffset, mask);

        // 计算 row col
        Reg::Div(diagRow, diagIdx, diagLenReg, mask); // dig_row = diag_idx / diag_len
        Reg::Muls(tmp, diagRow, diagLen, mask);
        Reg::Sub(diagCol, diagIdx, tmp, mask); // diag_col = diag_idx % diag_len

        if constexpr (direction == ExpandDirection::Horiz) {
            // 横向，旋转90°，右移得到 x 坐标
            // x_row = diag_col
            Reg::RegTensor<IdxType>& xRow = diagCol;
            // x_col = col_len - 1 - (diag_row + row_offset) + diag_col
            //       = diag_col - diag_row + (col_len - 1 - row_offset)
            Reg::Sub(tmp, diagCol, diagRow, mask);
            Reg::Adds(tmp, tmp, static_cast<RangeType>(colCnt - 1) - rowOffset, mask);
            Reg::RegTensor<IdxType>& xCol = tmp;

            // 有效值，小于0的会翻转
            // col_len 最大可能为 2^15，当 col_len 为最大值时，x_col 不可能为负
            // col_len 不为最大值时，x_col 翻转不可能小于 col_len
            Reg::Compares<IdxType, CMPMODE::LE>(validMask, xCol, static_cast<IdxType>(colCnt - 1), validMask);

            SimdDoScatter(x, diagReg, xRow, xCol, colCntReg, validMask);
        } else {
            // 纵向，直接上移
            // x_col = diag_col
            Reg::RegTensor<IdxType>& xCol = diagCol;
            // x_row = diag_row + diag_col + row_offset
            Reg::Adds(tmp, diagRow, rowOffset, mask);
            Reg::Add(tmp, tmp, diagCol, mask);
            Reg::RegTensor<IdxType>& xRow = tmp;

            // 有效值，小于0的会翻转
            // row_len 最大可能为 2^15，当 row_len 为最大值时，x_row 不可能为负
            // 当row_len 不为最大值时，x_row 翻转不可能小于 row_len
            Reg::Compares<IdxType, CMPMODE::LE>(validMask, xRow, static_cast<IdxType>(rowCnt - 1), validMask);

            SimdDoScatter(x, diagReg, xRow, xCol, colCntReg, validMask);
        }
    }
}

template <typename T, typename U>
template <ExpandDirection direction>
__simd_vf__ inline void MatrixSetDiagCutTail<T, U>::SimdSetDiagGather(__ubuf__ T* x, __ubuf__ T* diag, uint32_t diagLen,
                                                                      uint16_t loopNum, uint32_t processNum,
                                                                      uint32_t rowCnt, uint32_t colCnt,
                                                                      RangeType rowOffset, IdxType diagCnt,
                                                                      IdxType diagIdxOffset, IdxType ubAglinStart,
                                                                      IdxType ubAglinOffset)
{
    Reg::RegTensor<CastType, REG_TRAIT> diagReg;
    Reg::RegTensor<IdxType> xIdx;
    Reg::RegTensor<IdxType> xRow;
    Reg::RegTensor<IdxType> xCol;
    Reg::RegTensor<IdxType> diagRow;
    Reg::RegTensor<IdxType> diagIdx;
    Reg::RegTensor<IdxType> colCntReg;
    Reg::RegTensor<IdxType> diagLenReg;
    Reg::RegTensor<IdxType> tmp;
    Reg::RegTensor<IdxType> offsetReg;
    Reg::MaskReg mask;
    Reg::MaskReg validMask;
    Reg::MaskReg isLower;

    Reg::Duplicate(colCntReg, (IdxType)colCnt);
    Reg::Duplicate(diagLenReg, (IdxType)diagLen);
    IdxType lowerThreshold = ubAglinStart + diagIdxOffset;

    for (uint16_t i = 0; i < loopNum; i++) {
        mask = Reg::UpdateMask<MaskType>(processNum);
        // 1. Arange 生成 x 线性索引
        Reg::Arange((Reg::RegTensor<RangeType>&)xIdx, i * VREG_ELEMENTS);
        // 2. xRow = xIdx / colCnt, xCol = xIdx % colCnt
        Reg::Div(xRow, xIdx, colCntReg, mask);
        Reg::Muls(tmp, xRow, colCnt, mask);
        Reg::Sub(xCol, xIdx, tmp, mask);
        // 3. diagRow = rowOffset + xRow - xCol
        //    rowOffset = endK - upperLeftK，在 caller 侧用 int64 算好后截断到 RangeType，
        //    范围 [-(rowCnt-1), colCnt-1]，一定能放进 RangeType。
        //    unsigned 回绕保持模 2^N 正确：
        //      - k in [startK, endK] => diagRow in [0, diagCnt-1] (valid)
        //      - k > endK            => diagRow 回绕为大值          (invalid)
        //      - k < startK          => diagRow >= diagCnt          (invalid)
        Reg::Sub(diagRow, xRow, xCol, mask);
        Reg::Adds(diagRow, diagRow, rowOffset, mask);
        // 4. validMask = (diagRow < diagCnt) — 单条 unsigned 比较，替代原先
        //    (startK <= k <= endK) 两条带符号比较（k 回绕时必失效）
        Reg::Compares<IdxType, CMPMODE::LT>(validMask, diagRow, diagCnt, mask);
        // 5. diagIdx = diagRow * diagLen + diagCol
        if constexpr (direction == ExpandDirection::Horiz) {
            diagIdx = xRow;
        } else {
            diagIdx = xCol;
        }
        Reg::MulAddDst(diagIdx, diagRow, diagLenReg, mask);
        // 6. gatherIdx = diagIdx - diagIdxOffset + (ubAglinOffset if isLower)
        Reg::Compares<IdxType, CMPMODE::GE>(isLower, diagIdx, lowerThreshold, mask);
        Reg::Adds(diagIdx, diagIdx, -static_cast<IdxType>(diagIdxOffset), mask);
        Reg::Duplicate(offsetReg, ubAglinOffset, isLower);
        Reg::Add(diagIdx, diagIdx, offsetReg, mask);
        // 7. Gather + StoreAlign
        Reg::Gather(diagReg, diag, (Reg::RegTensor<IdxType>&)diagIdx, validMask);
        Reg::StoreAlign<CastType, STORE_DIST>((__ubuf__ CastType*)(x + i * VREG_ELEMENTS), diagReg, validMask);
    }
}

template <typename T, typename U>
template <ExpandDirection direction>
__simd_vf__ inline void MatrixSetDiagCutTail<T, U>::SimdSetDiagGatherScatter(
    __ubuf__ T* x, __ubuf__ T* diag, uint32_t diagLen, uint16_t loopNum, uint32_t processNum, uint32_t rowCnt,
    uint32_t colCnt, uint32_t diagCnt, RangeType colOffset, IdxType diagCntIdx, IdxType diagIdxOffset,
    IdxType ubAglinStart, IdxType ubAglinOffset)
{
    Reg::RegTensor<CastType, REG_TRAIT> diagReg;
    Reg::RegTensor<IdxType> pIdx;
    Reg::RegTensor<IdxType> pRow; // 平行四边形行，等于 xRow
    Reg::RegTensor<IdxType> pCol;
    Reg::RegTensor<IdxType> xCol;
    Reg::RegTensor<IdxType> xIdx;
    Reg::RegTensor<IdxType> diagRow;
    Reg::RegTensor<IdxType> diagIdx;
    Reg::RegTensor<IdxType> diagCntReg;
    Reg::RegTensor<IdxType> colCntReg;
    Reg::RegTensor<IdxType> diagLenReg;
    Reg::RegTensor<IdxType> tmp;
    Reg::RegTensor<IdxType> tmp2;
    Reg::RegTensor<IdxType> offsetReg;
    Reg::MaskReg mask;
    Reg::MaskReg validMask;
    // Reg::MaskReg geZero;
    // Reg::MaskReg ltColCnt;
    Reg::MaskReg isLower;

    Reg::Duplicate(diagCntReg, (IdxType)diagCnt);
    Reg::Duplicate(colCntReg, (IdxType)colCnt);
    Reg::Duplicate(diagLenReg, (IdxType)diagLen);
    IdxType lowerThreshold = ubAglinStart + diagIdxOffset;

    for (uint16_t i = 0; i < loopNum; i++) {
        mask = Reg::UpdateMask<MaskType>(processNum);
        // 1. Arange 生成平行四边形线性索引
        Reg::Arange((Reg::RegTensor<RangeType>&)pIdx, i * VREG_ELEMENTS);
        // 2. pRow = pIdx / diagCnt, pCol = pIdx % diagCnt
        Reg::Div(pRow, pIdx, diagCntReg, mask);
        Reg::Muls(tmp, pRow, diagCnt, mask);
        Reg::Sub(pCol, pIdx, tmp, mask);
        // 3. xCol = colOffset + pCol + pRow
        //    colOffset = startK - upperLeftK，在 caller 侧用 int64 算好后截断到 RangeType，
        //    范围 [-(rowCnt-1), colCnt-1]，一定能放进 RangeType。
        //    不再经过 k = startK + pCol，避免 int32 的 startK 在 IdxType=uint16 时截断。
        Reg::Adds(xCol, pCol, colOffset, mask);
        Reg::Add(xCol, xCol, pRow, mask);
        // 4. validMask = (0 <= xCol < colCnt)
        //    unsigned 下 xCol >= 0 恒真，但 xCol < 0 会回绕为大值被 xCol < colCnt 过滤掉。
        // Reg::Compares<IdxType, CMPMODE::GE>(geZero, xCol, (IdxType)0, mask);
        Reg::Compares<IdxType, CMPMODE::LT>(validMask, xCol, static_cast<IdxType>(colCnt), mask);
        // Reg::And(validMask, geZero, ltColCnt, mask);
        // 5. xIdx = pRow * colCnt + xCol  (pRow = xRow)
        Reg::Mul(xIdx, pRow, colCntReg, mask);
        Reg::Add(xIdx, xIdx, xCol, mask);
        // 6. diagRow = diagCntIdx - pCol (= (diagCnt-1) - pCol)
        //    pCol in [0, diagCnt-1] => diagRow in [0, diagCnt-1]，恒有效，无回绕。
        Reg::Duplicate(tmp2, diagCntIdx);
        Reg::Sub(diagRow, tmp2, pCol, mask);
        // 7. diagIdx = diagRow * diagLen + diagCol
        if constexpr (direction == ExpandDirection::Horiz) {
            diagIdx = pRow; // diagCol = pRow
        } else {
            diagIdx = xCol; // diagCol = xCol
        }
        Reg::MulAddDst(diagIdx, diagRow, diagLenReg, mask);
        // 8. gatherIdx = diagIdx - diagIdxOffset + (ubAglinOffset if isLower)
        Reg::Compares<IdxType, CMPMODE::GE>(isLower, diagIdx, lowerThreshold, mask);
        Reg::Adds(diagIdx, diagIdx, -static_cast<IdxType>(diagIdxOffset), mask);
        Reg::Duplicate(offsetReg, ubAglinOffset, isLower);
        Reg::Add(diagIdx, diagIdx, offsetReg, mask);
        // 9. Gather + Scatter
        Reg::Gather(diagReg, diag, (Reg::RegTensor<IdxType>&)diagIdx, validMask);
        Reg::Scatter(x, (Reg::RegTensor<T>&)diagReg, (Reg::RegTensor<IdxType>&)xIdx, validMask);
    }
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::ComputeByScatter(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                                                    MSDCutTailBlockInfo& info)
{
    // 搬入 diag
    uint64_t diagStart = info.batch * diagElementsPerBatch_ + static_cast<uint64_t>(k1_ - info.endK) * maxDiagLen_;
    uint32_t diagIdxOffset{0};
    uint32_t ubAglinStart{0};
    uint32_t ubAglinOffset{0};
    CopyInDiag<direction>(diagTensor, diagStart, info, diagIdxOffset, ubAglinStart, ubAglinOffset);
    // 要处理所有搬入的对角线 + ub block 对齐的空洞 - 偏移
    uint32_t totalCnt = info.diagSize + ubAglinOffset - diagIdxOffset;
    uint16_t loopNum = Ops::Base::CeilDiv(totalCnt, VREG_ELEMENTS);
    // 计算
    if constexpr (direction == ExpandDirection::Horiz) {
        // 最后一条对角线相对当前x右上角的偏移
        info.rowOffset = info.upperRightK - info.endK;
    } else {
        // 最后一条对角线相对当前x左上角的偏移
        info.rowOffset = info.upperLeftK - info.endK;
    }
    Mutex::Unlock<PIPE_MTE2>(info.eventID);
    Mutex::Lock<PIPE_V>(info.eventID);
    asc_vf_call<SimdSetDiagScatter<direction>>(
        (__ubuf__ T*)(xTensor.GetPhyAddr()), (__ubuf__ T*)(diagTensor.GetPhyAddr()), info.diagLen, loopNum, totalCnt,
        info.rowCnt, info.colCnt, info.rowOffset, diagIdxOffset, ubAglinStart, ubAglinOffset);
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::ComputeByGather(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                                                   MSDCutTailBlockInfo& info)
{
    // 搬入 diag（复用现有逻辑）
    uint64_t diagStart = info.batch * diagElementsPerBatch_ + static_cast<uint64_t>(k1_ - info.endK) * maxDiagLen_;
    uint32_t diagIdxOffset{0};
    uint32_t ubAglinStart{0};
    uint32_t ubAglinOffset{0};
    CopyInDiag<direction>(diagTensor, diagStart, info, diagIdxOffset, ubAglinStart, ubAglinOffset);
    // Gather 遍历 x 全量
    uint32_t totalCnt = info.rowCnt * info.colCnt;
    uint16_t loopNum = Ops::Base::CeilDiv(totalCnt, VREG_ELEMENTS);
    // rowOffset = endK - upperLeftK: caller 侧用 int64 算好后截断到 RangeType，
    // 范围 [-(rowCnt-1), colCnt-1]，安全放入 RangeType（int16/int32）
    RangeType rowOffset = static_cast<RangeType>(info.endK - info.upperLeftK);
    Mutex::Unlock<PIPE_MTE2>(info.eventID);
    Mutex::Lock<PIPE_V>(info.eventID);
    asc_vf_call<SimdSetDiagGather<direction>>((__ubuf__ T*)(xTensor.GetPhyAddr()),
                                              (__ubuf__ T*)(diagTensor.GetPhyAddr()), info.diagLen, loopNum, totalCnt,
                                              info.rowCnt, info.colCnt, rowOffset, static_cast<IdxType>(info.diagCnt),
                                              diagIdxOffset, ubAglinStart, ubAglinOffset);
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::ComputeByGatherScatter(LocalTensor<T>& xTensor,
                                                                          LocalTensor<T>& diagTensor,
                                                                          MSDCutTailBlockInfo& info)
{
    // 搬入 diag（复用现有逻辑）
    uint64_t diagStart = info.batch * diagElementsPerBatch_ + static_cast<uint64_t>(k1_ - info.endK) * maxDiagLen_;
    uint32_t diagIdxOffset{0};
    uint32_t ubAglinStart{0};
    uint32_t ubAglinOffset{0};
    CopyInDiag<direction>(diagTensor, diagStart, info, diagIdxOffset, ubAglinStart, ubAglinOffset);
    // Gather+Scatter 遍历平行四边形
    uint32_t totalCnt = info.rowCnt * info.diagCnt;
    uint16_t loopNum = Ops::Base::CeilDiv(totalCnt, VREG_ELEMENTS);
    // colOffset = startK - upperLeftK: caller 侧用 int64 算好后截断到 RangeType，
    // 范围 [-(rowCnt-1), colCnt-1]，安全放入 RangeType（int16/int32）
    RangeType colOffset = static_cast<RangeType>(info.startK - info.upperLeftK);
    Mutex::Unlock<PIPE_MTE2>(info.eventID);
    Mutex::Lock<PIPE_V>(info.eventID);
    asc_vf_call<SimdSetDiagGatherScatter<direction>>(
        (__ubuf__ T*)(xTensor.GetPhyAddr()), (__ubuf__ T*)(diagTensor.GetPhyAddr()), info.diagLen, loopNum, totalCnt,
        info.rowCnt, info.colCnt, info.diagCnt, colOffset, static_cast<IdxType>(info.diagCnt - 1), diagIdxOffset,
        ubAglinStart, ubAglinOffset);
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::ShouldUseSimt(const MSDCutTailBlockInfo& info) const
{
    // 特殊场景，完整尾轴，有下对角线，搬运逻辑无法处理，直接走simt
    if constexpr (direction == ExpandDirection::Horiz) {
        if (info.diagLen == maxDiagLen_ && info.startK < 0) {
            return true;
        }
    }
    // 对角线数据较小
    if (info.diagSize < SIMD_MIN_DIAG_SIZE) {
        return true;
    }
    // 对角线数据过于离散
    if (info.diagLen != maxDiagLen_ && info.diagLen < SIMD_MIN_DIAG_LEN) {
        return true;
    }
    // 默认不走 simt
    return false;
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::ShouldUseGather(const MSDCutTailBlockInfo& info) const
{
    // diag 元素数 >= x 元素数
    if (info.diagSize < info.rowCnt * info.colCnt) {
        return false;
    }
    return true;
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::ShouldUseGatherScatter(const MSDCutTailBlockInfo& info) const
{
    // 只支持横向
    if constexpr (direction == ExpandDirection::Vert) {
        return false;
    }
    // 小 dtype
    if constexpr (D_SIZE > GATHER_SCATTER_MAX_DSIZE) {
        return false;
    }
    return true;
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline bool MatrixSetDiagCutTail<T, U>::ShouldUseScatter(const MSDCutTailBlockInfo& info) const
{
    // 无 scatter bank 冲突
    return !IsScatterServeConflict(info);
}

template <typename T, typename U>
template <ExpandDirection direction>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::CopyInDiagAndCompute(LocalTensor<T>& xTensor,
                                                                        LocalTensor<T>& diagTensor,
                                                                        MSDCutTailBlockInfo& info)
{
    // 计算当前块对角线的长度
    info.diagCnt = info.endK - info.startK + 1;
    // 计算对角线块大小
    info.diagSize = info.diagLen * info.diagCnt;

    // simt
    if (ShouldUseSimt<direction>(info)) {
        ComputeBySimt(xTensor, diagTensor, info);
        return;
    }
    // gather
    if (!IsGatherServeConflict<direction>(info)) {
        // gather
        if (ShouldUseGather<direction>(info)) {
            ComputeByGather<direction>(xTensor, diagTensor, info);
            return;
        }
        // gather+scatter
        if (ShouldUseGatherScatter<direction>(info)) {
            ComputeByGatherScatter<direction>(xTensor, diagTensor, info);
            return;
        }
    }
    // scatter
    if (ShouldUseScatter<direction>(info)) {
        ComputeByScatter<direction>(xTensor, diagTensor, info);
        return;
    }
    // simt 兜底
    ComputeBySimt(xTensor, diagTensor, info);
}

/**
 * 计算左上角的k
 */
template <typename T, typename U>
__aicore__ inline int64_t MatrixSetDiagCutTail<T, U>::ComputeUpperLeftK(const MSDCutTailBlockInfo& info) const
{
    return static_cast<int64_t>(info.startCol) - static_cast<int64_t>(info.startRow);
}

/**
 * 计算右上角的k
 */
template <typename T, typename U>
__aicore__ inline int64_t MatrixSetDiagCutTail<T, U>::ComputeUpperRightK(const MSDCutTailBlockInfo& info) const
{
    return static_cast<int64_t>(info.endCol) - static_cast<int64_t>(info.startRow);
}

/**
 * 计算左下角的k
 */
template <typename T, typename U>
__aicore__ inline int64_t MatrixSetDiagCutTail<T, U>::ComputeBottomLeftK(const MSDCutTailBlockInfo& info) const
{
    return static_cast<int64_t>(info.startCol) - static_cast<int64_t>(info.endRow);
}

/**
 * 计算右下角的k
 */
template <typename T, typename U>
__aicore__ inline int64_t MatrixSetDiagCutTail<T, U>::ComputeBottomRightK(const MSDCutTailBlockInfo& info) const
{
    return static_cast<int64_t>(info.endCol) - static_cast<int64_t>(info.endRow);
}

template <typename T, typename U>
template <XTilingShape xTilingShape>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::Compute(LocalTensor<T>& xTensor, LocalTensor<T>& diagTensor,
                                                           MSDCutTailBlockInfo& info)
{
    if constexpr (xTilingShape == XTilingShape::HorizRect) {
        // 右边界，即右下角的对角线
        int64_t& rightBoardK = info.bottomRightK;
        if (info.startK > rightBoardK) {
            info.diagLen = info.rowCnt - (info.startK - rightBoardK);
            CopyInDiagAndCompute<ExpandDirection::Horiz>(xTensor, diagTensor, info);
            return;
        }
        // 左边界，即左上角的对角线
        int64_t& leftBoardK = info.upperLeftK;
        if (info.endK < leftBoardK) {
            info.diagLen = info.rowCnt - (leftBoardK - info.endK);
            CopyInDiagAndCompute<ExpandDirection::Vert>(xTensor, diagTensor, info);
            return;
        }
        // 中间
        info.diagLen = info.rowCnt;
        CopyInDiagAndCompute<ExpandDirection::Horiz>(xTensor, diagTensor, info);
        return;
    } else {
        // 上边界，即左上角的对角线
        int64_t& topBoardK = info.upperLeftK;
        if (info.startK > topBoardK) {
            info.diagLen = info.colCnt - (info.startK - topBoardK);
            CopyInDiagAndCompute<ExpandDirection::Horiz>(xTensor, diagTensor, info);
            return;
        }
        // 下边界，即右下角的对角线
        int64_t& bottomBoardK = info.bottomRightK;
        if (info.endK < bottomBoardK) {
            info.diagLen = info.colCnt - (bottomBoardK - info.endK);
            CopyInDiagAndCompute<ExpandDirection::Vert>(xTensor, diagTensor, info);
            return;
        }
        // 中间
        info.diagLen = info.colCnt;
        CopyInDiagAndCompute<ExpandDirection::Vert>(xTensor, diagTensor, info);
        return;
    }
}

template <typename T, typename U>
template <XTilingShape xTilingShape>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::ProcessWithXShape()
{
    // 参数计算
    uint64_t startIdx = blockIdx_ * ubPerCore_;
    uint64_t cntInCol = Ops::Base::CeilDiv(xColNum_, static_cast<uint64_t>(xColFactor_));
    uint64_t cntInRow = Ops::Base::CeilDiv(xRowNum_, static_cast<uint64_t>(xRowFactor_));
    uint64_t cntPerBatch = cntInRow * cntInCol;
    uint64_t totalCnt = batchSize_ * cntPerBatch;
    if (startIdx >= totalCnt) {
        return;
    }

    uint32_t endIdx = Std::min(startIdx + ubPerCore_, totalCnt);
    uint64_t batchElements = xRowNum_ * xColNum_;

    LocalTensor<T> xPing = buf_.Get<T>(X_MAX_ELEMENTS);
    LocalTensor<T> diagPing = buf_.GetWithOffset<T>(DIAG_MAX_ELEMENTS, X_MAX_BUF_SIZE);
    LocalTensor<T> xPong = buf_.GetWithOffset<T>(X_MAX_ELEMENTS, X_MAX_BUF_SIZE + DIAG_MAX_BUF_SIZE + BANK_OFFSET);
    LocalTensor<T> diagPong = buf_.GetWithOffset<T>(DIAG_MAX_ELEMENTS,
                                                    X_MAX_BUF_SIZE * 2 + DIAG_MAX_BUF_SIZE + BANK_OFFSET);

    MutexID mutexId0 = AllocMutexID();
    MutexID mutexId1 = AllocMutexID();

    // 遍历UB块
    for (uint64_t idx = startIdx; idx < endIdx; ++idx) {
        MSDCutTailBlockInfo info;
        // 同步事件
        info.eventID = (idx & 1) ? mutexId1 : mutexId0;
        // 计算当前块范围
        info.batch = idx / cntPerBatch;
        uint64_t hwIdx = idx - info.batch * cntPerBatch;
        info.startRow = (hwIdx / cntInCol) * xRowFactor_;
        info.rowCnt = Std::min(static_cast<uint64_t>(xRowFactor_), xRowNum_ - info.startRow);
        info.endRow = info.startRow + info.rowCnt - 1;
        info.startCol = (hwIdx % cntInCol) * xColFactor_;
        info.colCnt = Std::min(static_cast<uint64_t>(xColFactor_), xColNum_ - info.startCol);
        info.endCol = info.startCol + info.colCnt - 1;
        // 计算对角线范围
        info.upperLeftK = ComputeUpperLeftK(info);
        info.upperRightK = ComputeUpperRightK(info);
        info.bottomLeftK = ComputeBottomLeftK(info);
        info.bottomRightK = ComputeBottomRightK(info);
        int64_t startK = Std::max(static_cast<int64_t>(k0_), info.bottomLeftK);
        int64_t endK = Std::min(static_cast<int64_t>(k1_), info.upperRightK);
        // x偏移
        uint64_t xStart = info.batch * batchElements + info.startRow * xColNum_ + info.startCol;
        uint32_t processNum = info.rowCnt * info.colCnt;

        Mutex::Lock<PIPE_MTE2>(info.eventID);
        LocalTensor<T>& xTensor = (idx & 1) ? xPong : xPing;

        if (startK > endK) {
            // 当前块不存在对角线数据，直接搬入搬出
            CopyInX(xTensor, xStart, processNum);
            Mutex::Unlock<PIPE_MTE2>(info.eventID);
            Mutex::Lock<PIPE_MTE3>(info.eventID);
            CopyOut(xTensor, xStart, processNum);
        } else {
            info.startK = startK;
            info.endK = endK;
            // 搬入 X
            CopyInX(xTensor, xStart, processNum, info);
            // 处理对角线
            LocalTensor<T>& diagTensor = (idx & 1) ? diagPong : diagPing;
            Compute<xTilingShape>(xTensor, diagTensor, info);
            // 搬出
            Mutex::Unlock<PIPE_V>(info.eventID);
            Mutex::Lock<PIPE_MTE3>(info.eventID);
            CopyOut(xTensor, xStart, processNum);
        }
        Mutex::Unlock<PIPE_MTE3>(info.eventID);
    }
    ReleaseMutexID(mutexId0);
    ReleaseMutexID(mutexId1);
}

template <typename T, typename U>
__aicore__ inline void MatrixSetDiagCutTail<T, U>::Process()
{
    // 横向矩阵与纵向矩阵的simd处理策略不同
    if (xRowFactor_ < xColFactor_) {
        ProcessWithXShape<XTilingShape::HorizRect>();
    } else {
        ProcessWithXShape<XTilingShape::VertRect>();
    }
}
} // namespace MSD

#endif
