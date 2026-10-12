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
 * \file transpose_cut_one_axis.h
 * \brief TilingKey=10002 CUT_ONCE 策略实现
 *
 * 适用场景：2-5维 NDDMA 场景，DoSplitUB 后 outCutIndex ≤ FindOutIndex(inCutIndex)，
 * 即输出切分轴不比输入切分轴"更靠前"，只需切1个轴。
 *
 * 核心设计：
 *   - 使用 DataCopy<T, NDDMA_MAX_DIM_NUM> 5维 NDDMA 搬运，硬件自动完成维度重排
 *   - 单缓冲 TQueBind（BUFFER_NUM=1），UB 大小 = tiling_->ubSize
 *   - Main/Tail 分支：每个 block loop 判断是否为尾块，选择对应的 NDDMA 参数
 *
 * NDDMA 5维搬运的转置原理：
 *   DataCopy<T, 5, config> 通过 NDDMA 参数控制硬件5维跨步读取+重排写入：
 *   - loopSize[5]：每维循环次数
 *   - loopSrcStride[5]：源地址每维步长（以元素为单位）
 *   - loopDstStride[5]：目标地址每维步长（以元素为单位）
 *   硬件根据 loopSrcStride 从 GM 读取数据，按 loopDstStride 写入 UB，自动完成维度重排
 *
 * 混合基地址计算（DecimalToMixed）：
 *   将线性 block 索引 loopIdx 分解到各输出维度的混合基表示，
 *   再通过 expandedPerm 映射回输入维度，计算源地址偏移：
 *   srcOffset = Σ(mixedBase[i] * srcLoopStride[expandedPerm[i]])
 *
 * CopyOut 使用 LoopModeParams 处理最多5维的循环展开：
 *   - blockLen/blockCount/dstStride 处理前2维
 *   - loop1Size/loop1SrcStride/loop1DstStride 处理第3维
 *   - loop2Size/loop2SrcStride/loop2DstStride 处理第4维
 *   - 第5维(loop4)手动循环展开
 */
#ifndef TRANSPOSE_CUT_ONE_AXIS_H
#define TRANSPOSE_CUT_ONE_AXIS_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

/**
 * @brief CUT_ONCE 策略类，实现 NDDMA 5维搬运 + 单轴切分转置
 *
 * 数据流：GM → DataCopy<T,5>(UB, NDDMA自动转置) → DataCopyPad(GM, LoopMode展开写出)
 * 支持多核并行（ParseMultiCoreRange）和 Main/Tail 尾块处理。
 *
 * @tparam T 数据元素类型
 */
template <typename T>
class TransposeCutOneAxis : public TransposeBase<T> {
public:
    __aicore__ inline TransposeCutOneAxis(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline NdDmaLoopInfo<NDDMA_MAX_DIM_NUM> SetupLoopInfo(const int64_t inUbSrcShape[],
                                                                     const int64_t inUbDstShape[]);
    __aicore__ inline void ProcessMain(int64_t loopidxEnd);
    __aicore__ inline void ProcessTail();
    __aicore__ inline void GetLoopParams(int64_t n);
    __aicore__ inline void DecimalToMixed(int64_t num, int64_t bases[],
                                          int64_t mixedBase[]); ///< 十进制转混合基（地址计算核心）
    __aicore__ inline void GetLoopAndStride();                  ///< 计算源/目标循环步长
    __aicore__ inline void CopyIn(int64_t loopIdx, NdDmaParams<T, NDDMA_MAX_DIM_NUM>& params);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[],
                                   int64_t loopDstStride[]);
    __aicore__ inline void ProcessPerCore();

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_; // 单缓冲队列（CUT_ONCE 无需双缓冲）
    GlobalTensor<T> inputGM_;                                     // 输入 GM 张量
    GlobalTensor<T> outputGM_;                                    // 输出 GM 张量

    // tiling params
    const TransposeOpTilingData* tiling_;

    // core params
    int64_t blkProcessNum_ = 0;
    int64_t blkProcessIdxStart_ = 0;
    int64_t blkProcessIdxEnd_ = 0;

    // blockCount and blockLen of copyOut
    int64_t mainBlockLen_ = 1;
    int64_t tailBlockLen_ = 1;
    int64_t blockCount_ = 1;
    int64_t dstStrideTotalLen_ = 1;

    // expand output init
    int64_t expandedOutputCutIndex_; ///< 输出切分轴在5维扩展后的索引
    int64_t expandedInputCutIndex_;  ///< 输入切分轴在5维扩展后的索引（由 expandedPerm 映射得到）

    // address offset
    int64_t srcLoopSize_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};   ///< 源各维度的循环次数
    int64_t dstLoopSize_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};   ///< 目标各维度的循环次数
    int64_t srcLoopStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0}; ///< 源各维度的步长（元素数）
    int64_t dstLoopStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0}; ///< 目标各维度的步长（元素数）
    int64_t srcAddressOffsetMixedBase_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0}; ///< 源地址偏移的混合基分解结果
    int64_t dstAddressOffsetMixedBase_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0}; ///< 目标地址偏移的混合基分解结果

    // NDDMA DataCopy 参数（Main 和 Tail 各一组）
    int64_t loopSizeMain_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};      ///< Main 块 NDDMA loopSize
    int64_t loopSrcStrideMain_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0}; ///< Main 块 NDDMA loopSrcStride
    int64_t loopDstStrideMain_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0}; ///< Main 块 NDDMA loopDstStride
    int64_t loopSizeTail_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};      ///< Tail 块 NDDMA loopSize
    int64_t loopSrcStrideTail_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0}; ///< Tail 块 NDDMA loopSrcStride
    int64_t loopDstStrideTail_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0}; ///< Tail 块 NDDMA loopDstStride
};

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    ParseTilingData();
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, 1, tiling_->ubSize);
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::Process()
{
    if (!ParseMultiCoreRange(blockIdx_, tiling_->realCoreNum, tiling_->blkFactor, tiling_->blkTailFactor,
                             blkProcessNum_, blkProcessIdxStart_, blkProcessIdxEnd_)) {
        return;
    }
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::ParseTilingData()
{
    expandedOutputCutIndex_ = tiling_->outCutIndex + NDDMA_MAX_DIM_NUM - tiling_->permSize;
}

/**
 * @brief 构建 NDDMA 5维搬运的 LoopInfo 参数
 *
 * 根据 UB 内的源 shape（inUbSrcShape）和目标 shape（inUbDstShape），
 * 计算 NDDMA 硬件所需的 loopSize/loopSrcStride/loopDstStride 参数。
 *
 * 计算逻辑：
 * 1. loopSize[i] = inUbSrcShape[i]（每维搬运次数 = 源shape[i]）
 * 2. loopSrcStride[i] = Π(expandedInputShape[j], j>i)（源步长 = 后续维度乘积）
 * 3. loopDstStride 的计算较为复杂：
 *    a. 先按输出顺序计算临时步长 loopDstStrideTmp[i] = Π(inUbDstShape[j], j>i)
 *    b. 在输出切分轴前一个位置做 32字节对齐（因为 UB → GM 需要按 Block 对齐）
 *    c. 对齐后重新递推前面的步长
 *    d. 最后通过 expandedPerm 映射：loopDstStride[expandedPerm[i]] = loopDstStrideTmp[i]
 *       （NDDMA 硬件的 dstStride 按"输入维度的顺序"排列，所以需要 perm 映射）
 * 4. 所有数组逆序（NDDMA 硬件要求逆序排列）
 *
 * @param inUbSrcShape UB 内源 shape（5维）
 * @param inUbDstShape UB 内目标 shape（5维）
 * @return NDDMA 搬运参数
 */
template <typename T>
__aicore__ inline NdDmaLoopInfo<NDDMA_MAX_DIM_NUM> TransposeCutOneAxis<T>::SetupLoopInfo(const int64_t inUbSrcShape[],
                                                                                         const int64_t inUbDstShape[])
{
    NdDmaLoopInfo<NDDMA_MAX_DIM_NUM> loopInfo;
    int64_t loopDstStrideTmp[NDDMA_MAX_DIM_NUM] = {0};
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        int64_t recumMultiSrc = 1;
        int64_t recumMultiDst = 1;
        loopInfo.loopSize[i] = inUbSrcShape[i];
        for (int64_t j = i + 1; j < NDDMA_MAX_DIM_NUM; j++) {
            recumMultiSrc *= tiling_->expandedInputShape[j];
            recumMultiDst *= inUbDstShape[j];
        }
        loopInfo.loopSrcStride[i] = recumMultiSrc;
        loopDstStrideTmp[i] = recumMultiDst;
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
    loopInfo.loopSrcStride[NDDMA_MAX_DIM_NUM - 1] = 1;
    loopDstStrideTmp[NDDMA_MAX_DIM_NUM - 1] = 1;

    int64_t alignDstStride = 1;
    if (expandedOutputCutIndex_ - 1 >= 0) {
        // 在输出切分轴前一个位置做 32 字节对齐
        // 因为 UB → GM 的 DataCopyPad 要求 blockLen 按 BLOCK_SIZE_BYTE 对齐
        alignDstStride = (loopDstStrideTmp[expandedOutputCutIndex_ - 1] * sizeof(T) + BLOCK_SIZE_BYTE - 1) /
                         BLOCK_SIZE_BYTE;
        alignDstStride = alignDstStride * BLOCK_SIZE_BYTE / sizeof(T); // 转回元素单位
        loopDstStrideTmp[expandedOutputCutIndex_ - 1] = alignDstStride;
    }

    for (int64_t i = expandedOutputCutIndex_ - 2; i >= 0; i--) {
        loopDstStrideTmp[i] = inUbDstShape[i + 1] * loopDstStrideTmp[i + 1];
    }

    // NDDMA 硬件的 loopDstStride 按"输入维度的顺序"排列
    // expandedPerm[i] 表示输出第i维对应输入第 expandedPerm[i] 维
    // 因此 loopDstStride[expandedPerm[i]] = loopDstStrideTmp[i]
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        loopInfo.loopDstStride[tiling_->expandedPerm[i]] = loopDstStrideTmp[i];
    }
    // NDDMA 硬件要求参数逆序排列（dim4→dim0）
    this->reverseArray(loopInfo.loopSize);
    this->reverseArray(loopInfo.loopSrcStride);
    this->reverseArray(loopInfo.loopDstStride);

    return loopInfo;
}

/**
 * @brief 十进制数转混合基表示
 *
 * 将线性索引 num 分解到各维度的混合基表示，用于计算多维偏移地址。
 * 算法：从最低维（i=NDDMA_MAX_DIM_NUM-1）开始，依次取余和整除，
 * mixedBase[i] = num % bases[i]，num /= bases[i]。
 *
 * 示例（3维 shape=[2,3,4]，num=17）：
 *   bases = [2, 3, 4]（从高维到低维）
 *   mixedBase = [17%2=1, 8%3=2, 2%4=2] → 地址 = 1*stride0 + 2*stride1 + 2*stride2
 *
 * @param num       待分解的十进制数（线性索引）
 * @param bases     各维度的基数（循环次数数组）
 * @param mixedBase [out] 混合基分解结果
 */
template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[])
{
    if (num == 0) {
        return;
    }
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        mixedBase[i] = num % bases[i];
        num /= bases[i];
        if (num == 0) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::GetLoopParams(int64_t n)
{
    const int64_t reverseIdx = NDDMA_MAX_DIM_NUM - 1 - n;
    // main and tail loopSize
    loopSizeMain_[n] = tiling_->inUbMainDstShape[reverseIdx];
    loopSizeTail_[n] = tiling_->inUbTailDstShape[reverseIdx];
    // main loopSrcStride and loopDstStride
    if (n > 0) {
        if (n == NDDMA_MAX_DIM_NUM - expandedOutputCutIndex_) {
            loopSrcStrideMain_[n] = Ops::Base::CeilAlign(
                                        static_cast<int64_t>(tiling_->inUbMainDstShape[NDDMA_MAX_DIM_NUM - n] *
                                                             loopSrcStrideMain_[n - 1] * sizeof(T)),
                                        BLOCK_SIZE_BYTE) /
                                    sizeof(T);
        } else {
            loopSrcStrideMain_[n] = loopSrcStrideMain_[n - 1] * tiling_->inUbMainDstShape[NDDMA_MAX_DIM_NUM - n];
        }
        loopDstStrideMain_[n] = loopDstStrideMain_[n - 1] * tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - n];
    }
    // tail loopSrcStride and loopDstStride
    if (tiling_->outTailFactor != 0) {
        if (n > 0) {
            if (n == NDDMA_MAX_DIM_NUM - expandedOutputCutIndex_) {
                loopSrcStrideTail_[n] = Ops::Base::CeilAlign(
                                            static_cast<int64_t>(tiling_->inUbTailDstShape[NDDMA_MAX_DIM_NUM - n] *
                                                                 loopSrcStrideTail_[n - 1] * sizeof(T)),
                                            BLOCK_SIZE_BYTE) /
                                        sizeof(T);
            } else {
                loopSrcStrideTail_[n] = loopSrcStrideTail_[n - 1] * tiling_->inUbTailDstShape[NDDMA_MAX_DIM_NUM - n];
            }
            loopDstStrideTail_[n] = loopDstStrideTail_[n - 1] * tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - n];
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::GetLoopAndStride()
{
    // calculate the loopSize and stride of src and dst
    expandedInputCutIndex_ = tiling_->expandedPerm[expandedOutputCutIndex_];
    srcLoopStride_[NDDMA_MAX_DIM_NUM - 1] = 1;
    dstLoopStride_[NDDMA_MAX_DIM_NUM - 1] = 1;
    srcLoopStride_[NDDMA_MAX_DIM_NUM - 2] = tiling_->expandedInputShape[NDDMA_MAX_DIM_NUM - 1];
    dstLoopStride_[NDDMA_MAX_DIM_NUM - 2] = tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - 1];
    for (int64_t i = NDDMA_MAX_DIM_NUM - 3; i >= 0; i--) {
        srcLoopStride_[i] = srcLoopStride_[i + 1] * tiling_->expandedInputShape[i + 1];
        dstLoopStride_[i] = dstLoopStride_[i + 1] * tiling_->expandedOutputShape[i + 1];
    }
    srcLoopStride_[expandedInputCutIndex_] *= tiling_->outUbFactor;
    dstLoopStride_[expandedOutputCutIndex_] *= tiling_->outUbFactor;

    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        srcLoopSize_[i] = Ops::Base::CeilDiv(tiling_->expandedInputShape[i], tiling_->inUbMainSrcShape[i]);
        dstLoopSize_[i] = Ops::Base::CeilDiv(tiling_->expandedOutputShape[i], tiling_->inUbMainDstShape[i]);
    }
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::CopyIn(int64_t loopIdx, NdDmaParams<T, NDDMA_MAX_DIM_NUM>& params)
{
    int64_t srcAddressOffset = 0;
    DecimalToMixed(loopIdx, dstLoopSize_, dstAddressOffsetMixedBase_);
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        srcAddressOffsetMixedBase_[tiling_->expandedPerm[i]] = dstAddressOffsetMixedBase_[i];
        srcAddressOffset += srcAddressOffsetMixedBase_[tiling_->expandedPerm[i]] *
                            srcLoopStride_[tiling_->expandedPerm[i]];
    }
    LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
    DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], params);
    vecQue_.EnQue(bindLocalIn);
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::CopyOut(int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[],
                                                       int64_t loopDstStride[])
{
    int64_t dstAddressOffset = 0;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        dstAddressOffset += dstAddressOffsetMixedBase_[i] * dstLoopStride_[i];
    }
    // MTE3 parms
    DataCopyExtParams copyOutParams{1, 0, 0, 0, 0};
    LoopModeParams loopParams;

    copyOutParams.blockLen = sizeof(T);
    int64_t endIndex = 0;
    int64_t dstStride = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        copyOutParams.blockLen *= loopSize[NDDMA_MAX_DIM_NUM - 1 - i];
        if (tiling_->expandedOutputShape[i] != tiling_->inUbMainDstShape[i]) {
            endIndex = NDDMA_MAX_DIM_NUM - 1 - i;
            break;
        }
    }
    if (endIndex + 1 < NDDMA_MAX_DIM_NUM) {
        copyOutParams.blockCount = loopSize[endIndex + 1];
        copyOutParams.dstStride = copyOutParams.blockCount == 1 ?
                                      0 :
                                      (loopDstStride[endIndex + 1] - copyOutParams.blockLen / sizeof(T)) * sizeof(T);
    }
    loopParams.loop1Size = 1;
    if (endIndex + 2 < NDDMA_MAX_DIM_NUM) {
        loopParams.loop1Size = loopSize[endIndex + 2];
        loopParams.loop1SrcStride = loopSrcStride[endIndex + 2] * sizeof(T);
        loopParams.loop1DstStride = loopDstStride[endIndex + 2] * sizeof(T);
    }
    loopParams.loop2Size = 1;
    if (endIndex + 3 < NDDMA_MAX_DIM_NUM) {
        loopParams.loop2Size = loopSize[endIndex + 3];
        loopParams.loop2SrcStride = loopSrcStride[endIndex + 3] * sizeof(T);
        loopParams.loop2DstStride = loopDstStride[endIndex + 3] * sizeof(T);
    }
    LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
    if (endIndex + 4 < NDDMA_MAX_DIM_NUM) {
        for (int64_t loop4Idx = 0; loop4Idx < loopSize[4]; loop4Idx++) {
            SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
            DataCopyPad(outputGM_[dstAddressOffset + loop4Idx * loopDstStride[4]],
                        bindLocalOut[loop4Idx * loopSrcStride[4]], copyOutParams);
            ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        }
    } else {
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(outputGM_[dstAddressOffset], bindLocalOut, copyOutParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
    vecQue_.FreeTensor(bindLocalOut);
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::ProcessPerCore()
{
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        GetLoopParams(i);
    }
    GetLoopAndStride();

    // MTE2 params main
    T constValue = 0;
    NdDmaLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoMain = SetupLoopInfo(tiling_->inUbMainSrcShape, tiling_->inUbMainDstShape);
    NdDmaParams<T, NDDMA_MAX_DIM_NUM> paramsMain = {loopInfoMain, constValue};

    int64_t outCutLoopSize = Ops::Base::CeilDiv(tiling_->expandedOutputShape[expandedOutputCutIndex_],
                                                tiling_->outUbFactor);
    for (int64_t loopIdx = blkProcessIdxStart_; loopIdx < blkProcessIdxEnd_; loopIdx++) {
        if (tiling_->outTailFactor != 0 && (loopIdx + 1) % outCutLoopSize == 0) { // tail
            // MTE2 params tail
            NdDmaLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoTail = SetupLoopInfo(tiling_->inUbTailSrcShape,
                                                                          tiling_->inUbTailDstShape);
            NdDmaParams<T, NDDMA_MAX_DIM_NUM> paramsTail = {loopInfoTail, constValue};
            CopyIn(loopIdx, paramsTail);
            CopyOut(loopIdx, loopSizeTail_, loopSrcStrideTail_, loopDstStrideTail_);
        } else {
            CopyIn(loopIdx, paramsMain);
            CopyOut(loopIdx, loopSizeMain_, loopSrcStrideMain_, loopDstStrideMain_);
        }
    }
}
} // namespace Transpose

#endif // TRANSPOSE_CUT_ONE_AXIS
