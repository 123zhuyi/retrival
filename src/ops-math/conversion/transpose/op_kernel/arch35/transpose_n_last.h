/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_n_last.h
 * \brief TilingKey=10004 N_LAST_TRANSPOSE 策略实现
 *
 * 适用场景：尾轴不转置（reducedPerm[dim-1]==dim-1）且尾轴大小 ≥ 32（MOVEALIGN_LAST_MIN_ELE），
 * 即输入的最后一维在输出中保持连续，可以按连续行进行搬移。
 *
 * 核心优势：
 *   - 尾轴连续意味着无需 NDDMA 转置，直接按行搬运即可
 *   - 使用双缓冲流水线（BUFFER_NUM=2），CopyIn 和 CopyOut 可重叠执行
 *   - 支持8维循环展开（loop4~7），提高 CopyOut 效率
 *   - 混合基地址计算（GetDstAddressOffset）将线性 loopIdx 分解到各输出维度，
 *     再按 perm 映射回输入维度计算偏移
 *
 * 性能优化标记（已在代码中实现）：
 *   P2: main/tail 循环分离，消除热路径分支
 *   P4: 缓存数组成员到局部变量，避免别名导致的重载
 *   P6: 缓存 tiling 指针字段到局部变量，避免双重 Load
 *   P9: 融合 DecimalToMixed + 累加，无中间数组，无零初始化开销
 */
#ifndef TRANSPOSE_N_LAST_H
#define TRANSPOSE_N_LAST_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

/**
 * @brief N_LAST_TRANSPOSE 策略类，实现尾轴不转置的连续行搬移
 *
 * 利用尾轴连续的特性，按行搬运数据，双缓冲流水线加速。
 * 支持8维循环展开，适用于 2-8 维输入。
 *
 * @tparam T 数据元素类型
 */
template <typename T>
class TransposeNLast : public TransposeBase<T> {
public:
    __aicore__ inline TransposeNLast(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void GetBlockLoopNum();
    __aicore__ inline void GetLoopParams(int64_t n);
    __aicore__ inline int64_t GetDstAddressOffset(int64_t loopIdx, int64_t permSize, const int64_t mixedBase[],
                                                  const int64_t loopNum[]);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void CopyIn(int64_t loopIdx, int64_t inputBlockLen, int64_t inCutLoopSize, int64_t inUbFactor,
                                  int64_t inTailFactor, int64_t permSize);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[],
                                   int64_t loopDstStride[], int64_t permSize, const int64_t mixedBase[],
                                   const int64_t loopNum[]);

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_; // 双缓冲绑定队列（BUFFER_NUM=2）
    GlobalTensor<T> inputGM_;                                     // 输入 GM 张量
    GlobalTensor<T> outputGM_;                                    // 输出 GM 张量

    // tiling params
    const TransposeOpTilingData* tiling_;
    int64_t reducedInputShape_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};   ///< 简化后的输入 shape
    int64_t reducedOutputShape_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};  ///< 简化后的输出 shape
    int64_t inUbInputShapeMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};  ///< Main 块输入 UB shape
    int64_t inUbOutputShapeMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< Main 块输出 UB shape
    int64_t inUbInputShapeTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};  ///< Tail 块输入 UB shape
    int64_t inUbOutputShapeTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< Tail 块输出 UB shape
    int64_t outputBlockLenMain_ = 0; ///< Main 块输出尾轴长度（连续元素数）
    int64_t outputBlockLenTail_ = 0; ///< Tail 块输出尾轴长度

    // core params
    int64_t blkProcessNum_ = 0;
    int64_t blkProcessIdxStart_ = 0;
    int64_t blkProcessIdxEnd_ = 0;

    // 地址偏移参数（用于 GetDstAddressOffset 的混合基分解）
    int64_t dstAddressOffsetMixedBase_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< 输出地址偏移混合基
    int64_t dstLoopNumArray_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};           ///< 输出各维度步长

    // DataCopy 参数：Main 和 Tail 各一组
    int64_t loopSizeMain_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};      ///< Main 块各维 loopSize
    int64_t loopSrcStrideMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< Main 块各维 srcStride
    int64_t loopDstStrideMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< Main 块各维 dstStride
    int64_t loopSizeTail_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};      ///< Tail 块各维 loopSize
    int64_t loopSrcStrideTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< Tail 块各维 srcStride
    int64_t loopDstStrideTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0}; ///< Tail 块各维 dstStride
};

template <typename T>
__aicore__ inline void TransposeNLast<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                               TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    ParseTilingData();
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, BUFFER_NUM, tiling_->ubSize / BUFFER_NUM);
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::Process()
{
    if (!ParseMultiCoreRange(blockIdx_, tiling_->realCoreNum, tiling_->blkFactor, tiling_->blkTailFactor,
                             blkProcessNum_, blkProcessIdxStart_, blkProcessIdxEnd_)) {
        return;
    }
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::ParseTilingData()
{
    // input in ub shape main and tail
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        reducedInputShape_[i] = tiling_->inputShape[i];
        if (i > tiling_->inCutIndex) {
            inUbInputShapeMain_[i] = reducedInputShape_[i];
            inUbInputShapeTail_[i] = reducedInputShape_[i];
        } else {
            inUbInputShapeMain_[i] = 1;
            inUbInputShapeTail_[i] = 1;
        }
    }
    inUbInputShapeMain_[tiling_->inCutIndex] = tiling_->inUbFactor;
    inUbInputShapeTail_[tiling_->inCutIndex] = tiling_->inTailFactor;

    // output in ub shape main and tail
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        reducedOutputShape_[i] = reducedInputShape_[tiling_->perm[i]];
        inUbOutputShapeMain_[i] = inUbInputShapeMain_[tiling_->perm[i]];
        inUbOutputShapeTail_[i] = inUbInputShapeTail_[tiling_->perm[i]];
    }

    outputBlockLenMain_ = inUbOutputShapeMain_[tiling_->permSize - 1];
    outputBlockLenTail_ = inUbOutputShapeTail_[tiling_->permSize - 1];
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::GetBlockLoopNum()
{
    const int64_t permSize = tiling_->permSize;
    int64_t dstLoopNumArrayTmp[TRANSPOSE_MAX_AXIS_NUM];
    int64_t rightProducts[TRANSPOSE_MAX_AXIS_NUM + 1];
    rightProducts[permSize] = 1;

    for (int64_t i = permSize - 1; i >= 0; i--) {
        rightProducts[i] = rightProducts[i + 1] * reducedOutputShape_[i];
    }
    for (int64_t i = 0; i < permSize; i++) {
        dstAddressOffsetMixedBase_[i] = Ops::Base::CeilDiv(reducedInputShape_[i], inUbInputShapeMain_[i]);
        dstLoopNumArrayTmp[i] = inUbOutputShapeMain_[i] * rightProducts[i + 1];
    }
    for (int64_t i = 0; i < permSize; i++) {
        dstLoopNumArray_[tiling_->perm[i]] = dstLoopNumArrayTmp[i];
    }
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::GetLoopParams(int64_t n)
{
    const int64_t permSize = tiling_->permSize;
    const int64_t reverseIdx = permSize - 1 - n;
    // main and tail loopSize
    loopSizeMain_[n] = inUbInputShapeMain_[reverseIdx];
    loopSizeTail_[n] = inUbInputShapeTail_[reverseIdx];
    // main and tail last axis aligned to ub block size
    int64_t alignedStrideMain = Ops::Base::CeilAlign(static_cast<int64_t>(loopSizeMain_[0] * sizeof(T)),
                                                     BLOCK_SIZE_BYTE) /
                                sizeof(T);
    int64_t alignedStrideTail = Ops::Base::CeilAlign(static_cast<int64_t>(loopSizeTail_[0] * sizeof(T)),
                                                     BLOCK_SIZE_BYTE) /
                                sizeof(T);
    // main loopSrcStride and loopDstStride
    if (loopSizeMain_[n] != 1) {
        loopSrcStrideMain_[n] = alignedStrideMain;
        if (n == 0) {
            loopSrcStrideMain_[n] = 1;
        }
        for (int64_t i = 1; i < n; i++) {
            loopSrcStrideMain_[n] *= inUbInputShapeMain_[permSize - 1 - i];
        }
        loopDstStrideMain_[n] = 1;
        for (int64_t i = permSize - 1; i >= 0; i--) {
            if (tiling_->perm[i] != reverseIdx) {
                loopDstStrideMain_[n] *= reducedOutputShape_[i];
            } else {
                break;
            }
        }
    }
    // tail loopSrcStride and loopDstStride
    if (tiling_->inTailFactor != 0 && loopSizeTail_[n] != 1) {
        loopSrcStrideTail_[n] = alignedStrideTail;
        for (int64_t i = 1; i < n; i++) {
            loopSrcStrideTail_[n] *= inUbInputShapeTail_[permSize - 1 - i];
        }
        loopDstStrideTail_[n] = 1;
        for (int64_t i = permSize - 1; i >= 0; i--) {
            if (tiling_->perm[i] != reverseIdx) {
                loopDstStrideTail_[n] *= reducedOutputShape_[i];
            } else {
                break;
            }
        }
    }
}

/**
 * @brief 计算目标地址偏移（融合 DecimalToMixed + 累加，P9 优化）
 *
 * 将线性 loopIdx 分解到各输出维度，然后累加计算目标地址偏移。
 * 直接在循环中执行混合基分解和累加，避免创建中间数组和零初始化开销。
 *
 * 算法：
 *   for i = permSize-1 downto 0:
 *     dstAddressOffset += (num % mixedBase[i]) * loopNum[i]
 *     num /= mixedBase[i]
 *
 * @param loopIdx  当前循环索引
 * @param permSize 维度数
 * @param mixedBase 混合基基数数组（各维度的循环次数）
 * @param loopNum   各维度的地址步长
 * @return 目标地址偏移（元素数）
 */
template <typename T>
__aicore__ inline int64_t TransposeNLast<T>::GetDstAddressOffset(int64_t loopIdx, int64_t permSize,
                                                                 const int64_t mixedBase[], const int64_t loopNum[])
{
    int64_t dstAddressOffset = 0;
    if (loopIdx != 0) {
        int64_t num = loopIdx;
        for (int64_t i = permSize - 1; i >= 0 && num != 0; i--) {
            dstAddressOffset += (num % mixedBase[i]) * loopNum[i];
            num /= mixedBase[i];
        }
    }
    return dstAddressOffset;
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::CopyIn(int64_t loopIdx, int64_t inputBlockLen, int64_t inCutLoopSize,
                                                 int64_t inUbFactor, int64_t inTailFactor, int64_t permSize)
{
    int64_t inputBlockLenMain = inputBlockLen * inUbFactor;
    int64_t inputBlockLenTail = inputBlockLen * inTailFactor;
    int64_t srcAddressOffset = loopIdx * inputBlockLenMain;
    if (inTailFactor != 0) {
        srcAddressOffset = (loopIdx - loopIdx / inCutLoopSize) * inputBlockLenMain +
                           loopIdx / inCutLoopSize * inputBlockLenTail;
    }
    int64_t blockLen = inputBlockLenMain;
    int64_t lastAxisLen = inUbInputShapeMain_[permSize - 1];
    if (inTailFactor != 0 && (loopIdx + 1) % inCutLoopSize == 0) {
        blockLen = inputBlockLenTail;
        lastAxisLen = inUbInputShapeTail_[permSize - 1];
    }
    LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
    DataCopyExtParams copyInParams{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    copyInParams.blockLen = blockLen * sizeof(T);
    copyInParams.blockCount = 1;
    if ((lastAxisLen * sizeof(T)) % BLOCK_SIZE_BYTE != 0) {
        copyInParams.blockLen = lastAxisLen * sizeof(T);
        copyInParams.blockCount = blockLen / lastAxisLen;
    }
    DataCopyPad(bindLocalIn, inputGM_[srcAddressOffset], copyInParams, padParams);
    vecQue_.EnQue(bindLocalIn);
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::CopyOut(int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[],
                                                  int64_t loopDstStride[], int64_t permSize, const int64_t mixedBase[],
                                                  const int64_t loopNum[])
{
    int64_t dstAddressOffset = GetDstAddressOffset(loopIdx, permSize, mixedBase, loopNum);

    // P4: cache array elements into locals to avoid alias-induced reload after DataCopyPad stores
    const int64_t ls4 = loopSize[4], ls5 = loopSize[5], ls6 = loopSize[6], ls7 = loopSize[7];
    const int64_t lss4 = loopSrcStride[4], lss5 = loopSrcStride[5];
    const int64_t lss6 = loopSrcStride[6], lss7 = loopSrcStride[7];
    const int64_t lds4 = loopDstStride[4], lds5 = loopDstStride[5];
    const int64_t lds6 = loopDstStride[6], lds7 = loopDstStride[7];

    DataCopyExtParams copyOutParams{1, 0, 0, 0, 0};
    copyOutParams.blockLen = loopSize[0] * sizeof(T);
    copyOutParams.blockCount = loopSize[1];
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = copyOutParams.blockCount == 1 ? 0 : (loopDstStride[1] - loopSize[0]) * sizeof(T);

    LoopModeParams loopParams;
    loopParams.loop1Size = loopSize[2];
    loopParams.loop1SrcStride = loopSrcStride[2] * sizeof(T);
    loopParams.loop1DstStride = loopDstStride[2] * sizeof(T);
    loopParams.loop2Size = loopSize[3];
    loopParams.loop2SrcStride = loopSrcStride[3] * sizeof(T);
    loopParams.loop2DstStride = loopDstStride[3] * sizeof(T);

    LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
    // Move SetLoopModePara outside the loop: loopParams is constant across all iterations
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    for (int64_t loop7Idx = 0; loop7Idx < ls7; loop7Idx++) {
        for (int64_t loop6Idx = 0; loop6Idx < ls6; loop6Idx++) {
            for (int64_t loop5Idx = 0; loop5Idx < ls5; loop5Idx++) {
                for (int64_t loop4Idx = 0; loop4Idx < ls4; loop4Idx++) {
                    DataCopyPad(outputGM_[dstAddressOffset + loop7Idx * lds7 + loop6Idx * lds6 + loop5Idx * lds5 +
                                          loop4Idx * lds4],
                                bindLocalOut[loop7Idx * lss7 + loop6Idx * lss6 + loop5Idx * lss5 + loop4Idx * lss4],
                                copyOutParams);
                }
            }
        }
    }
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    vecQue_.FreeTensor(bindLocalOut);
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::ProcessPerCore()
{
    // P6: cache tiling pointer fields into local variables to avoid repeated double-Load
    const int64_t permSize = tiling_->permSize;
    const int64_t inUbFactor = tiling_->inUbFactor;
    const int64_t inTailFactor = tiling_->inTailFactor;
    const int64_t inCutIndex = tiling_->inCutIndex;

    for (int64_t i = 0; i < permSize; i++) {
        GetLoopParams(i);
    }
    GetBlockLoopNum();
    int64_t inputBlockLen = 1;
    for (int64_t i = permSize - 1; i > inCutIndex; i--) {
        inputBlockLen *= reducedInputShape_[i];
    }
    int64_t inCutLoopSize = Ops::Base::CeilDiv(reducedInputShape_[inCutIndex], inUbFactor);

    // P4: cache member variables used only in this function
    const int64_t loopStart = blkProcessIdxStart_;
    const int64_t loopEnd = blkProcessIdxEnd_;

    // P1: cache member arrays to stack to break alias pollution from struct dynamic indexing
    int64_t localMixedBase[TRANSPOSE_MAX_AXIS_NUM];
    int64_t localLoopNum[TRANSPOSE_MAX_AXIS_NUM];
    for (int64_t i = 0; i < permSize; i++) {
        localMixedBase[i] = dstAddressOffsetMixedBase_[i];
        localLoopNum[i] = dstLoopNumArray_[i];
    }

    // P2: main/tail loop separation to eliminate per-iteration branch in hot path
    for (int64_t loopIdx = loopStart; loopIdx < loopEnd; loopIdx++) {
        if (inTailFactor != 0 && (loopIdx + 1) % inCutLoopSize == 0) {
            CopyIn(loopIdx, inputBlockLen, inCutLoopSize, inUbFactor, inTailFactor, permSize);
            CopyOut(loopIdx, loopSizeTail_, loopSrcStrideTail_, loopDstStrideTail_, permSize, localMixedBase,
                    localLoopNum);
        } else {
            CopyIn(loopIdx, inputBlockLen, inCutLoopSize, inUbFactor, inTailFactor, permSize);
            CopyOut(loopIdx, loopSizeMain_, loopSrcStrideMain_, loopDstStrideMain_, permSize, localMixedBase,
                    localLoopNum);
        }
    }
}
} // namespace Transpose

#endif // TRANSPOSE_N_LAST_H
