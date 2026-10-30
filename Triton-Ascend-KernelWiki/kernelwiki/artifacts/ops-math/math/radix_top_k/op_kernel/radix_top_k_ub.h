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
 * \file radix_top_k_ub.h
 * \brief Radix TopK kernel — UB-local variant (tileHist & tileTopK in UB)
 */

#ifndef RADIX_TOPK_UB_H
#define RADIX_TOPK_UB_H

#include "radix_top_k_common.h"

namespace RadixTopK {
using namespace AscendC;

template <typename T, bool largest>
class RadixTopKUb : public RadixTopKBaseKernel<T, largest> {
    using Base = RadixTopKBaseKernel<T, largest>;

public:
    __aicore__ inline RadixTopKUb(TPipe& tpipe, const RadixTopKTilingData& tilingData) : Base(tpipe, tilingData) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace)
    {
        InitParams();
        this->xGm_.SetGlobalBuffer((__gm__ T*)x);
        this->valueGm_.SetGlobalBuffer((__gm__ T*)values);
        this->kGm_.SetGlobalBuffer((__gm__ int32_t*)k);
        this->indexGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        // 当 k 大于 workspace 所需空间时复用最后一个 batch 的 indices 内存
        __gm__ int32_t* wsPtr = this->tiling_.needWorkspace ?
                                    ((__gm__ int32_t*)workspace) :
                                    ((__gm__ int32_t*)indices + (this->batch_ - 1) * this->kValue_);
        this->globalHistGm_.SetGlobalBuffer(wsPtr);
        this->boundaryBinCumSumGm_.SetGlobalBuffer(wsPtr + this->numValue_ * this->coreNum_);
        this->coreTopKGm_.SetGlobalBuffer(wsPtr + this->numValue_ * this->coreNum_ + this->coreNum_);
        InitBuffers();
    }

    __aicore__ inline void SubProcess(uint64_t batchId);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void ClearTileTopK();
    __aicore__ inline void ClearHist(const int32_t& roundId);
    __aicore__ inline bool Update(const int32_t& roundId);
    __aicore__ inline uint32_t CreateVecIndex4TopK();
    __aicore__ inline void TileTopK(const uint64_t& batchId, const uint64_t& blockOffset);

    __aicore__ inline void AddTileHistToTileTopK(LocalTensor<int32_t>& tileTopK, LocalTensor<int32_t>& tileHist,
                                                 LocalTensor<int32_t>& globalHist);
    __aicore__ inline void HandleLastRoundBoundary(LocalTensor<int32_t>& tileHist, LocalTensor<int32_t>& tileTopK,
                                                   LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);

    uint32_t tempBufSize_;                 /**< 临时 buffer 大小（max(tileHistSize, cmpMaskSize)） */
    TBuf<TPosition::VECCALC> tileTopKBuf_; /**< tileTopK 计数器 UB buffer */
    TBuf<TPosition::VECCALC> tempBuf_;     /**< 临时 buffer（复用为 tileHist / cmpMask / vecIndex） */
};

/**
 * @brief 主流程入口：遍历所有 batch，对每个 batch 完成 Radix TopK 计算
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::Process()
{
    for (int32_t batchId = 0; batchId < this->batch_; batchId++) {
        SubProcess(batchId);
        SyncAll();
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);
}

/**
 * @brief 初始化当前 core 的参数（直接调用基类 InitBaseParams）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::InitParams()
{
    this->InitBaseParams();
}

/**
 * @brief 初始化 UB 缓冲区：xBuf Ping-Pong、tileTopKBuf、tempBuf、globalHistBuf、outBuf
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::InitBuffers()
{
    this->tpipe_.InitBuffer(this->xBufPing_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(this->xBufPong_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(tileTopKBuf_, this->tileNumAlign_ * sizeof(int32_t));
    uint32_t tileHistSize = this->numValue_ * this->tileNumAlign_ * sizeof(int32_t);
    uint32_t cmpMaskSize = this->tileLen_ * sizeof(uint8_t) / BYTE_SIZE;
    tempBufSize_ = AscendC::Std::max(tileHistSize, cmpMaskSize);
    this->tpipe_.InitBuffer(tempBuf_, tempBufSize_);

    this->tpipe_.InitBuffer(this->globalHistBuf_, this->numValue_ * sizeof(int32_t));
    this->tpipe_.InitBuffer(this->outValueBuf_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(this->outIndexBuf_, this->tileLen_ * sizeof(int32_t));
}

/**
 * @brief 初始化 tileTopK 计数器
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::ClearTileTopK()
{
    LocalTensor<int32_t> tileTopK = tileTopKBuf_.Get<int32_t>();
    Duplicate<int32_t>(tileTopK, 0, this->tileNum_);
}

/**
 * @brief 清除/准备当前轮的直方图
 *        首轮：用 tileLen 初始化 tileHist（每个 tile 元素数 = tileLen）
 *        非首轮：从上一轮的 tileHist 中提取边界 bin 的数据（或做减法）
 * @param roundId 当前轮 ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::ClearHist(const int32_t& roundId)
{
    LocalTensor<int32_t> tileHist = tempBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = this->globalHistBuf_.template Get<int32_t>();
    PipeBarrier<PIPE_V>();
    if (roundId == this->round_ - 1) {
        Duplicate<int32_t>(tileHist, this->tileLen_, this->tileNum_);
        if (this->tailTileLen_ != this->tileLen_) {
            VToSSync();
            tileHist.SetValue(this->tileNum_ - 1, this->tailTileLen_);
        }
        this->globalHistBoundaryNum_ = (this->tileNum_ - 1) * this->tileLen_ + this->tailTileLen_;
    } else {
        if (this->boundaryBin < this->numValue_ - 1) {
            Sub(tileHist, tileHist[this->boundaryBin * this->tileNumAlign_],
                tileHist[this->boundaryBinPrev * this->tileNumAlign_], this->tileNum_);
        } else {
            CopyData<int32_t>(tileHist, tileHist[this->boundaryBin * this->tileNumAlign_], this->tileNum_);
        }
    }
    MTE3ToSSync();
    globalHist.SetValue(0, this->globalHistBoundaryNum_);
    PipeBarrier<PIPE_V>();
    SToVSync();
    Duplicate<int32_t>(tileHist[this->tileNumAlign_], 0, (this->numValue_ - 1) * this->tileNumAlign_);
}

/**
 * @brief 对单个 batch 执行完整 Radix TopK 流程：
 *        逐轮（8→1）Ping-Pong CopyIn → Twiddle → 直方图统计 → 全局同步 → 边界查找 → 累加 tileTopK
 *        最后调用 TileTopK 提取最终结果
 * @param batchId 当前 batch ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::SubProcess(uint64_t batchId)
{
    this->involvedMask16_ = 0;
    this->andMask16_ = 0xC000; // ((1 << BITS_PER_ROUND) - 1) << (16 - BITS_PER_ROUND);
    this->totalDefinitelyInTopK_ = 0;
    this->boundaryBin = 0;
    this->boundaryBinPrev = 1;
    this->globalHistBoundaryNum_ = this->tileNum_ * this->tileLen_;
    this->remainK_ = this->kValue_;
    uint64_t blockOffset = this->CalcBlockOffset(batchId);
    ClearTileTopK();

    for (int32_t roundId = this->round_ - 1; roundId >= 0; roundId--) {
        uint64_t firstTileLen = this->tileNum_ > 1 ? this->tileLen_ : this->tailTileLen_;
        this->CopyIn(this->xBufPing_, firstTileLen, blockOffset);
        this->pingPongFlag = true;
        this->template CopyOut2Ws<true>(this->numValue_, this->blockIdx_ * this->numValue_);
        ClearHist(roundId);

        SetFlag<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);

        for (int32_t tileId = 0; tileId < this->tileNum_; tileId++) {
            PINGPONG_TILE_BEGIN(tileId, blockOffset);

            this->NegateDataForLargest(curBuf, curTileLen);
            this->TwiddleInB16(curBuf, curTileLen);
            this->DoAndMask(curBuf, curTileLen);
            LocalTensor<int32_t> tileHist = tempBuf_.Get<int32_t>();
            this->CalcCumsumHistogram16(curBuf, roundId, tileHist, tileId, static_cast<int32_t>(this->tileNumAlign_), 0,
                                        curTileLen);

            PINGPONG_TILE_END(tileId);
        }
        this->template CopyOut2Ws<false>(this->numValue_, this->blockIdx_ * this->numValue_);
        SyncAll();
        // 找到topk，提前退出
        if (Update(roundId))
            break;
        this->andMask16_ >>= BITS_PER_ROUND;
    }
    SyncAll();
    TileTopK(batchId, blockOffset);
}

/**
 * @brief 单轮 Update：读入全局直方图 → 归约 → 找边界 bin → 累加确定性 TopK → 处理末轮边界
 * @param roundId 当前轮 ID
 * @return true: 可提前退出（remainK_==0 或 roundId==0）
 */
template <typename T, bool largest>
__aicore__ inline bool RadixTopKUb<T, largest>::Update(const int32_t& roundId)
{
    LocalTensor<int32_t> tileHist = tempBuf_.Get<int32_t>();
    LocalTensor<int32_t> resTensor = this->outIndexBuf_.template Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;

    this->CopyInGlobalHist(globalHist);
    MTE2ToVSync();
    this->ReduceGlobalHist(globalHist);
    this->FindBoundaryBin(globalHist, roundId);

    LocalTensor<int32_t> tileTopK = tileTopKBuf_.Get<int32_t>();
    AddTileHistToTileTopK(tileTopK, tileHist, globalHist);

    this->remainK_ = this->kValue_ - this->totalDefinitelyInTopK_;
    LocalTensor<float> tileHistFp32 = resTensor.template ReinterpretCast<float>();
    if (roundId == 0 && this->remainK_ > 0) {
        HandleLastRoundBoundary(tileHist, tileTopK, resTensor, tileHistFp32);
    }

    bool isLastRound = (this->remainK_ == 0 || roundId == 0);
    // coreTopKGm_ 仅在 TileTopK 阶段被 CopyInCoreTopK 读取，只需在最终轮写一次
    if (isLastRound) {
        // tileNum_ 可能达数千，分段 ReduceSum 保证 fp32 累加不丢失 int32 精度
        int32_t totalTileTopKInCore = 0;
        int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
        int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
        int32_t segSize = this->tileNum_ < safeSumTileNum ? this->tileNum_ : safeSumTileNum;
        for (int32_t segOff = 0; segOff < this->tileNum_; segOff += segSize) {
            int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(this->tileNum_ - segOff));
            PipeBarrier<PIPE_V>();
            SToVSync();
            Cast(tileHistFp32, tileTopK[segOff], RoundMode::CAST_NONE, curSegLen);
            PipeBarrier<PIPE_V>();
            ReduceSum(tileHistFp32, tileHistFp32, tileHistFp32, curSegLen);
            VToSSync();
            totalTileTopKInCore += static_cast<int32_t>(tileHistFp32.GetValue(0));
        }
        resTensor.SetValue(0, totalTileTopKInCore);
        SToMTE3Sync();
        DataCopyExtParams coreTopKParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0, 0, 0};
        DataCopyPad(this->coreTopKGm_[this->blockIdx_], resTensor, coreTopKParams);
    }

    return isLastRound;
}

/**
 * @brief 将边界 bin 之前（即确定性在 TopK 中）的元素累加到 tileTopK 计数器
 * @param tileTopK 各 tile 的 TopK 计数
 * @param tileHist 各 tile 的直方图
 * @param globalHist 全局直方图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::AddTileHistToTileTopK(LocalTensor<int32_t>& tileTopK,
                                                                      LocalTensor<int32_t>& tileHist,
                                                                      LocalTensor<int32_t>& globalHist)
{
    if (globalHist(this->boundaryBin) == this->remainK_) {
        PipeBarrier<PIPE_V>();
        Add(tileTopK, tileTopK, tileHist[this->boundaryBin * this->tileNumAlign_], this->tileNum_);
        this->totalDefinitelyInTopK_ += globalHist(this->boundaryBin);
    } else if (this->boundaryBinPrev < this->numValue_) {
        PipeBarrier<PIPE_V>();
        Add(tileTopK, tileTopK, tileHist[this->boundaryBinPrev * this->tileNumAlign_], this->tileNum_);
        this->totalDefinitelyInTopK_ += globalHist(this->boundaryBinPrev);
    }
}

/**
 * @brief 处理最后一轮边界 bin 的元素分配：
 *        计算各 tile 边界 bin 元素数，按 core/tile 顺序将剩余 remainK_ 分配到各 tile 的 tileTopK
 * @param tileHist 各 tile 的直方图
 * @param tileTopK 各 tile 的 TopK 计数（会被更新）
 * @param resTensor 临时 Tensor
 * @param tileHistFp32 tileHist 的 fp32 视图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::HandleLastRoundBoundary(LocalTensor<int32_t>& tileHist,
                                                                        LocalTensor<int32_t>& tileTopK,
                                                                        LocalTensor<int32_t>& resTensor,
                                                                        LocalTensor<float>& tileHistFp32)
{
    PipeBarrier<PIPE_V>();
    if (this->boundaryBinPrev < this->numValue_) {
        Sub(resTensor, tileHist[this->boundaryBin * this->tileNumAlign_],
            tileHist[this->boundaryBinPrev * this->tileNumAlign_], this->tileNum_);
        PipeBarrier<PIPE_V>();
        Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, this->tileNum_);
    } else {
        Cast(tileHistFp32, tileHist[this->boundaryBin * this->tileNumAlign_], RoundMode::CAST_NONE, this->tileNum_);
    }
    int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
    int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
    int32_t segSize = this->tileNum_ < safeSumTileNum ? this->tileNum_ : safeSumTileNum;
    int32_t reduceSumValue = 0;
    PipeBarrier<PIPE_V>();
    for (int32_t segOff = 0; segOff < this->tileNum_; segOff += segSize) {
        int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(this->tileNum_ - segOff));
        SToVSync();
        ReduceSum(tileHistFp32, tileHistFp32[segOff], tileHistFp32[segOff], curSegLen);
        VToSSync();
        reduceSumValue += static_cast<int32_t>(tileHistFp32.GetValue(0));
    }
    resTensor.SetValue(0, reduceSumValue);
    SToMTE3Sync();
    this->CopyOutBoundaryBinCumSum(resTensor);
    SyncAll();

    int32_t cumSumValuePrev = 0;
    this->ComputeCumSumPrev(resTensor, cumSumValuePrev);
    int32_t remainCoreBoundaryNum = this->remainK_ - cumSumValuePrev;
    if (remainCoreBoundaryNum > 0) {
        for (int32_t tileId = 0; tileId < this->tileNum_; tileId++) {
            int32_t curTileBoundaryNum = tileHist[this->boundaryBin * this->tileNumAlign_].GetValue(tileId);
            if (this->boundaryBinPrev < this->numValue_) {
                curTileBoundaryNum -= tileHist[this->boundaryBinPrev * this->tileNumAlign_].GetValue(tileId);
            }
            if (curTileBoundaryNum < remainCoreBoundaryNum) {
                remainCoreBoundaryNum -= curTileBoundaryNum;
                tileTopK.SetValue(tileId, tileTopK.GetValue(tileId) + curTileBoundaryNum);
            } else {
                tileTopK.SetValue(tileId, tileTopK.GetValue(tileId) + remainCoreBoundaryNum);
                remainCoreBoundaryNum = 0;
                break;
            }
        }
    }
}

/**
 * @brief 为 TileTopK 阶段创建预生成索引，UB 变体从 tempBuf_ 剩余空间分配
 * @return 实际可用的 maxIndexLen
 */
template <typename T, bool largest>
__aicore__ inline uint32_t RadixTopKUb<T, largest>::CreateVecIndex4TopK()
{
    uint32_t cmpMaskSize = this->tileLen_ * sizeof(uint8_t) / BYTE_SIZE;
    uint32_t maxIndexLen = Ops::Base::FloorDiv(tempBufSize_ - cmpMaskSize, BLOCK_SIZE) * BLOCK_SIZE / sizeof(int32_t);
    if (this->tileLen_ < maxIndexLen) {
        maxIndexLen = this->tileLen_;
    } else if (maxIndexLen < 512) {
        maxIndexLen = 0;
    }
    if (maxIndexLen > 0) {
        LocalTensor<int32_t> vecIndex = tempBuf_.GetWithOffset<int32_t>(maxIndexLen, cmpMaskSize);
        Base::CreateVecIndex4TopK(maxIndexLen, vecIndex);
    }
    return maxIndexLen;
}

/**
 * @brief TileTopK 阶段：根据 tileTopK 计数从各 tile 提取最终 TopK 元素并写回输出
 *        UB 变体：tileTopK 在 tileTopKBuf_ 中，无需从 WS 分段拷贝
 * @param batchId 当前 batch ID
 * @param blockOffset 当前 core 在输入数据中的起始偏移
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKUb<T, largest>::TileTopK(const uint64_t& batchId, const uint64_t& blockOffset)
{
    LocalTensor<int32_t> tileTopK = tileTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> tmpLocal = tempBuf_.Get<int32_t>();
    uint64_t coreTopKOffset = this->CopyInCoreTopK(tmpLocal);
    uint64_t batchIndexOffset = batchId * this->kValue_;
    coreTopKOffset += batchIndexOffset;

    uint64_t firstId = 0;
    while (firstId < this->tileNum_ && tileTopK.GetValue(firstId) <= 0)
        firstId++;
    if (firstId >= this->tileNum_)
        return;

    uint64_t firstLen = (firstId == this->tileNum_ - 1) ? this->tailTileLen_ : this->tileLen_;
    this->CopyIn(this->xBufPing_, firstLen, blockOffset + firstId * this->tileLen_);
    uint32_t maxIndexLen = CreateVecIndex4TopK();

    this->InitTopKEvent();

    for (uint64_t tileId = firstId; tileId < this->tileNum_; tileId++) {
        PINGPONG_TILE_BEGIN(tileId, blockOffset);

        uint64_t tileBaseOffset = blockOffset - batchId * this->sortLen_;
        uint64_t tileOffset = tileBaseOffset + tileId * this->tileLen_;

        int32_t curTileK = tileTopK.GetValue(tileId);
        PipeBarrier<PIPE_V>();
        PROCESS_ONE_TILE_TOPK_OR_SKIP(
            curTileK, tempBuf_.GetWithOffset<int32_t>(maxIndexLen, this->tileLen_ * sizeof(uint8_t) / BYTE_SIZE));

        PINGPONG_TILE_END(tileId);
    }
    this->FinishTopKEvent();
}

} // namespace RadixTopK
#endif // RADIX_TOPK_UB_H
