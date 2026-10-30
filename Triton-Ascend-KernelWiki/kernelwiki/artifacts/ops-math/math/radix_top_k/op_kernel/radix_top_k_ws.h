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
 * \file radix_top_k_ws.h
 * \brief Radix TopK kernel — workspace variant (tileHist & tileTopK in GM workspace)
 */

#ifndef RADIX_TOPK_WS_H
#define RADIX_TOPK_WS_H

#include "radix_top_k_common.h"

namespace RadixTopK {
using namespace AscendC;

template <typename T, bool largest>
class RadixTopKWs : public RadixTopKBaseKernel<T, largest> {
    using Base = RadixTopKBaseKernel<T, largest>;

public:
    __aicore__ inline RadixTopKWs(TPipe& tpipe, const RadixTopKTilingData& tilingData) : Base(tpipe, tilingData) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace)
    {
        InitParams();
        this->xGm_.SetGlobalBuffer((__gm__ T*)x);
        this->valueGm_.SetGlobalBuffer((__gm__ T*)values);
        this->kGm_.SetGlobalBuffer((__gm__ int32_t*)k);
        this->indexGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        this->globalHistGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);
        this->boundaryBinCumSumGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_);
        this->coreTopKGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_ +
                                          this->coreNum_);
        uint32_t tileTopKOffset = 0;
        if (this->blockIdx_ < this->formerCoreNum_) {
            tileTopKOffset = this->blockIdx_ * this->formerTileNum_;
        } else {
            tileTopKOffset = this->formerCoreNum_ * this->formerTileNum_ +
                             (this->blockIdx_ - this->formerCoreNum_) * this->tailTileNum_;
        }
        uint32_t tileHistOffset = this->numValue_ * tileTopKOffset;
        tileTopKGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_ + this->coreNum_ * 2 +
                                    tileTopKOffset);
        tileHistGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + this->numValue_ * this->coreNum_ + this->coreNum_ * 2 +
                                    this->totalTileNum_ + tileHistOffset);
        InitBuffers();
    }

    __aicore__ inline void SubProcess(uint64_t batchId);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void ClearTileTopKInWs();
    __aicore__ inline void ClearHistInWs(const int32_t& roundId);
    __aicore__ inline void ClearTileHistInWs(const uint32_t& dstOffset, const uint32_t& dataNum);
    __aicore__ inline void CopyTileHistWs2Ub(const LocalTensor<int32_t>& dstTensor, const uint32_t& srcOffset,
                                             const uint32_t& dataNum);
    __aicore__ inline void CopyTileTopKWs2Ub(const LocalTensor<int32_t>& dstTensor, const uint32_t& srcOffset,
                                             const uint32_t& dataNum);
    __aicore__ inline void CopyTileTopKUb2Ws(const LocalTensor<int32_t>& srcTensor, const uint32_t& dstOffset,
                                             const uint32_t& dataNum);
    __aicore__ inline void CopyTileHistInWs(const uint32_t& dstOffset, const uint32_t& srcOffset);
    __aicore__ inline void SubTileHistWs2Ub(const LocalTensor<int32_t>& dstTensor,
                                            const LocalTensor<int32_t>& srcTensor, const uint32_t& srcOffset0,
                                            const uint32_t& srcOffset1, const uint32_t& dataNum);
    __aicore__ inline void SubTileHistInWs(const uint32_t& dstOffset, const uint32_t& srcOffset0,
                                           const uint32_t& srcOffset1, const uint32_t& dataNum);
    __aicore__ inline void SubTileHistInWsAll(const uint32_t& dstOffset, const uint32_t& srcOffset0,
                                              const uint32_t& srcOffset1);
    __aicore__ inline void AddTileHist2TileTopKInWs(const uint64_t& tileHistOffset);
    __aicore__ inline bool Update(const int32_t& roundId);
    __aicore__ inline uint32_t CreateVecIndex4TopK();
    __aicore__ inline uint64_t FindFirstTileWithTopK(const LocalTensor<int32_t>& tileTopK);
    __aicore__ inline void TileTopK(const uint64_t& batchId, const uint64_t& blockOffset);

    __aicore__ inline void AddTileHistToTileTopK(LocalTensor<int32_t>& globalHist);
    __aicore__ inline void HandleLastRoundBoundary(LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);
    __aicore__ inline void WriteCoreTopKFromWs(LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);

    GlobalTensor<int32_t> tileTopKGm_;            /**< WS 中 tileTopK 段的 Global Tensor */
    GlobalTensor<int32_t> tileHistGm_;            /**< WS 中 tileHist 段的 Global Tensor */
    uint32_t tileNumRepeatTimes_;                 /**< tile 写入 WS 的整块次数 */
    uint32_t tileNumRemain_;                      /**< tile 写入 WS 的剩余数量 */
    uint32_t tileNumBy2RepeatTimes_;              /**< tile 双 buffer 整块次数 */
    uint32_t tileNumBy2Remain_;                   /**< tile 双 buffer 剩余数量 */
    uint32_t tileNumBy3RepeatTimes_;              /**< tile 三 buffer 整块次数 */
    uint32_t tileNumBy3Remain_;                   /**< tile 三 buffer 剩余数量 */
    TBuf<TPosition::VECCALC> tileHistAndTopKBuf_; /**< tileHist 和 tileTopK 复用 UB buffer */
    TBuf<TPosition::VECCALC> tempBuf_;            /**< 临时 buffer（复用为 cmpMask / vecIndex） */
};

#include "radix_top_k_ws_utils.h"

/**
 * @brief 主流程入口：遍历所有 batch，对每个 batch 完成 WS 变体 Radix TopK 计算
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::Process()
{
    for (int32_t batchId = 0; batchId < this->batch_; batchId++) {
        SubProcess(batchId);
        SyncAll();
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);
}

/**
 * @brief 初始化基本参数（调用基类）+ WS 变体特有参数（tileNum 分段次数）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::InitParams()
{
    this->InitBaseParams();
    tileNumRepeatTimes_ = this->tileNum_ / MAX_TILE_NUM_IN_UB;
    tileNumRemain_ = this->tileNum_ % MAX_TILE_NUM_IN_UB;
    tileNumBy2RepeatTimes_ = this->tileNum_ / MAX_TILE_NUM_IN_UB_BY2;
    tileNumBy2Remain_ = this->tileNum_ % MAX_TILE_NUM_IN_UB_BY2;
    tileNumBy3RepeatTimes_ = this->tileNum_ / MAX_TILE_NUM_IN_UB_BY3;
    tileNumBy3Remain_ = this->tileNum_ % MAX_TILE_NUM_IN_UB_BY3;
}

/**
 * @brief 初始化 UB 缓冲区：xBuf Ping-Pong、tileHistAndTopKBuf、tempBuf、outBuf、globalHistBuf
 *        WS 变体中 tileHist/tileTopK 存 GM，UB 只用作分段读写缓冲
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::InitBuffers()
{
    uint32_t cmpMaskSize = this->tileLen_ * sizeof(uint8_t) / BYTE_SIZE;
    this->tpipe_.InitBuffer(this->xBufPing_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(this->xBufPong_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(tileHistAndTopKBuf_, MAX_TILE_NUM_IN_UB * sizeof(int32_t));
    this->tpipe_.InitBuffer(tempBuf_, cmpMaskSize);
    this->tpipe_.InitBuffer(this->outValueBuf_, this->tileLen_ * sizeof(float));
    this->tpipe_.InitBuffer(this->outIndexBuf_, this->tileLen_ * sizeof(int32_t));
    this->tpipe_.InitBuffer(this->globalHistBuf_, this->numValue_ * sizeof(int32_t));
}

/**
 * @brief 对单个 batch 执行 WS 变体 Radix TopK：
 *        逐轮 Ping-Pong CopyIn → Twiddle → 直方图统计(分段写 WS) → 全局同步 → 边界查找 → 累加 tileTopK(WS)
 * @param batchId 当前 batch ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::SubProcess(uint64_t batchId)
{
    this->involvedMask16_ = 0;
    this->andMask16_ = 0xC000; // ((1 << BITS_PER_ROUND) - 1) << (16 - BITS_PER_ROUND);
    this->totalDefinitelyInTopK_ = 0;
    this->boundaryBin = 0;
    this->boundaryBinPrev = 1;
    this->globalHistBoundaryNum_ = this->tileNum_ * this->tileLen_;
    this->remainK_ = this->kValue_;
    uint64_t blockOffset = this->CalcBlockOffset(batchId);
    ClearTileTopKInWs();

    for (int32_t roundId = this->round_ - 1; roundId >= 0; roundId--) {
        uint64_t firstTileLen = this->tileNum_ > 1 ? this->tileLen_ : this->tailTileLen_;
        this->CopyIn(this->xBufPing_, firstTileLen, blockOffset);
        this->pingPongFlag = true;
        this->template CopyOut2Ws<true>(this->numValue_, this->blockIdx_ * this->numValue_);
        ClearHistInWs(roundId);

        SetFlag<HardEvent::MTE2_V>(this->eventIDMTE2ToVForX_);

        for (int32_t repeatId = 0; repeatId <= tileNumBy3RepeatTimes_; repeatId++) {
            int32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB_BY3;
            int32_t endTileId = AscendC::Std::min(static_cast<int32_t>(startTileId + MAX_TILE_NUM_IN_UB_BY3),
                                                  static_cast<int32_t>(this->tileNum_));
            MTE3ToSSync();
            for (int32_t tileId = startTileId; tileId < endTileId; tileId++) {
                PINGPONG_TILE_BEGIN(tileId, blockOffset);

                this->NegateDataForLargest(curBuf, curTileLen);
                this->TwiddleInB16(curBuf, curTileLen);
                this->DoAndMask(curBuf, curTileLen);
                LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
                this->CalcCumsumHistogram16(curBuf, roundId, tileHist, tileId - startTileId, MAX_TILE_NUM_IN_UB_BY3,
                                            MAX_TILE_NUM_IN_UB_BY3, curTileLen);

                PINGPONG_TILE_END(tileId);
            }
            SToMTE3Sync();
            LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
            for (int32_t binMask = this->numValue_ - 1; binMask > 0; binMask--) {
                DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1),
                                                    static_cast<uint32_t>(sizeof(int32_t) * (endTileId - startTileId)),
                                                    0, 0, 0};
                uint32_t tileHistOffset = binMask * this->tileNum_ + repeatId * MAX_TILE_NUM_IN_UB_BY3;
                DataCopyPad(tileHistGm_[tileHistOffset], tileHist[(binMask - 1) * MAX_TILE_NUM_IN_UB_BY3],
                            tileHistOutParams);
            }
        }
        this->template CopyOut2Ws<false>(this->numValue_, this->blockIdx_ * this->numValue_);
        SyncAll();
        this->andMask16_ >>= BITS_PER_ROUND;
        if (Update(roundId))
            break;
    }
    SyncAll();
    TileTopK(batchId, blockOffset);
}

/**
 * @brief 单轮 Update（WS 变体）：读入全局直方图 → 归约 → 找边界 bin → 累加 tileTopK(WS) → 处理末轮边界 → 写 coreTopK
 * @param roundId 当前轮 ID
 * @return true: 可提前退出
 */
template <typename T, bool largest>
__aicore__ inline bool RadixTopKWs<T, largest>::Update(const int32_t& roundId)
{
    LocalTensor<int32_t> resTensor = this->outIndexBuf_.template Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;

    this->CopyInGlobalHist(globalHist);
    MTE2ToVSync();
    this->ReduceGlobalHist(globalHist);
    this->FindBoundaryBin(globalHist, roundId);

    AddTileHistToTileTopK(globalHist);

    this->remainK_ = this->kValue_ - this->totalDefinitelyInTopK_;
    LocalTensor<float> tileHistFp32 = resTensor.template ReinterpretCast<float>();
    if (roundId == 0 && this->remainK_ > 0) {
        HandleLastRoundBoundary(resTensor, tileHistFp32);
    }

    bool isLastRound = (this->remainK_ == 0 || roundId == 0);
    if (isLastRound) {
        WriteCoreTopKFromWs(resTensor, tileHistFp32);
    }

    return isLastRound;
}

/**
 * @brief WS 变体：将边界 bin 之前的确定性元素通过 WS 累加到 tileTopK
 * @param globalHist 全局直方图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::AddTileHistToTileTopK(LocalTensor<int32_t>& globalHist)
{
    if (globalHist(this->boundaryBin) == this->remainK_) {
        AddTileHist2TileTopKInWs(this->boundaryBin * this->tileNum_);
        this->totalDefinitelyInTopK_ += globalHist(this->boundaryBin);
    } else if (this->boundaryBinPrev < this->numValue_) {
        AddTileHist2TileTopKInWs(this->boundaryBinPrev * this->tileNum_);
        this->totalDefinitelyInTopK_ += globalHist(this->boundaryBinPrev);
    }
}

/**
 * @brief WS 变体：处理最后一轮边界 bin 的元素分配
 *        从 WS 读取各 tile 边界 bin 数据，计算每个 core 的元素分配
 * @param resTensor 临时 Tensor
 * @param tileHistFp32 tileHist 的 fp32 视图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::HandleLastRoundBoundary(LocalTensor<int32_t>& resTensor,
                                                                        LocalTensor<float>& tileHistFp32)
{
    uint32_t reduceSumValue = 0;
    for (int32_t i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = i == tileNumBy2RepeatTimes_ ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;
        if (this->boundaryBinPrev < this->numValue_) {
            SubTileHistWs2Ub(resTensor, resTensor, this->boundaryBin * this->tileNum_ + i * MAX_TILE_NUM_IN_UB_BY2,
                             this->boundaryBinPrev * this->tileNum_ + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
            PipeBarrier<PIPE_V>();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
        } else {
            CopyTileHistWs2Ub(resTensor, this->boundaryBin * this->tileNum_ + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
            MTE2ToVSync();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
        }
        int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
        int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
        int32_t segSize = dataNum < safeSumTileNum ? dataNum : safeSumTileNum;
        PipeBarrier<PIPE_V>();
        for (int32_t segOff = 0; segOff < dataNum; segOff += segSize) {
            int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(dataNum - segOff));
            SToVSync();
            ReduceSum(tileHistFp32, tileHistFp32[segOff], tileHistFp32[segOff], curSegLen);
            VToSSync();
            reduceSumValue += static_cast<int32_t>(tileHistFp32.GetValue(0));
        }
    }
    resTensor.SetValue(0, reduceSumValue);
    SToMTE3Sync();
    this->CopyOutBoundaryBinCumSum(resTensor);
    SyncAll();

    int32_t cumSumValuePrev = 0;
    this->ComputeCumSumPrev(resTensor, cumSumValuePrev);
    int32_t remainCoreBoundaryNum = this->remainK_ - cumSumValuePrev;
    if (remainCoreBoundaryNum > 0) {
        LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
        LocalTensor<int32_t> tileHistBoundary = tileTopK[MAX_TILE_NUM_IN_UB_BY3];
        LocalTensor<int32_t> tileHistBoundaryPrev = tileTopK[2 * MAX_TILE_NUM_IN_UB_BY3];
        for (int32_t i = 0; i <= tileNumBy3RepeatTimes_; i++) {
            uint32_t dataNum = i == tileNumBy3RepeatTimes_ ? tileNumBy3Remain_ : MAX_TILE_NUM_IN_UB_BY3;
            if (dataNum == 0)
                continue;
            uint32_t startTileId = i * MAX_TILE_NUM_IN_UB_BY3;
            uint32_t endTileId = startTileId + dataNum;

            CopyTileHistWs2Ub(tileHistBoundary, this->boundaryBin * this->tileNum_ + startTileId, dataNum);
            CopyTileHistWs2Ub(tileHistBoundaryPrev, this->boundaryBinPrev * this->tileNum_ + startTileId, dataNum);
            CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
            MTE2ToSSync();
            for (int32_t tileId = startTileId; tileId < endTileId; tileId++) {
                int32_t curTileBoundaryNum = tileHistBoundary.GetValue(tileId - startTileId);
                if (this->boundaryBinPrev < this->numValue_) {
                    curTileBoundaryNum -= tileHistBoundaryPrev.GetValue(tileId - startTileId);
                }
                if (curTileBoundaryNum < remainCoreBoundaryNum) {
                    remainCoreBoundaryNum -= curTileBoundaryNum;
                    tileTopK.SetValue(tileId - startTileId,
                                      tileTopK.GetValue(tileId - startTileId) + curTileBoundaryNum);
                } else {
                    tileTopK.SetValue(tileId - startTileId,
                                      tileTopK.GetValue(tileId - startTileId) + remainCoreBoundaryNum);
                    remainCoreBoundaryNum = 0;
                    break;
                }
            }
            SToMTE3Sync();
            CopyTileTopKUb2Ws(tileTopK, startTileId, dataNum);
            MTE3ToMTE2Sync();
        }
    }
}

/**
 * @brief WS 变体：从 WS 分段读取 tileTopK 累加得到本 core 总 TopK 计数并写回 WS
 * @param resTensor 临时 Tensor
 * @param tileHistFp32 tileHist 的 fp32 视图
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::WriteCoreTopKFromWs(LocalTensor<int32_t>& resTensor,
                                                                    LocalTensor<float>& tileHistFp32)
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    int32_t totalTileTopKInCore = 0;
    int32_t int32Align = BLOCK_SIZE / sizeof(int32_t);
    int32_t safeSumTileNum = FLOAT32_SAFE_INT / this->tileLen_ / int32Align * int32Align;
    for (int32_t i = 0; i <= tileNumRepeatTimes_; i++) {
        int32_t dataNum = i == tileNumRepeatTimes_ ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToVSync();
        int32_t segSize = dataNum < safeSumTileNum ? dataNum : safeSumTileNum;
        for (int32_t segOff = 0; segOff < dataNum; segOff += segSize) {
            int32_t curSegLen = AscendC::Std::min(segSize, static_cast<int32_t>(dataNum - segOff));
            SToVSync();
            Cast(tileHistFp32, tileTopK[segOff], RoundMode::CAST_NONE, curSegLen);
            PipeBarrier<PIPE_V>();
            ReduceSum(tileHistFp32, tileHistFp32, tileHistFp32, curSegLen);
            VToSSync();
            totalTileTopKInCore += static_cast<int32_t>(tileHistFp32.GetValue(0));
        }
    }
    this->CopyOutCoreTopK(resTensor, totalTileTopKInCore);
}

/**
 * @brief 为 TileTopK 阶段创建预生成索引，WS模板从 tileHistAndTopKBuf_ 剩余空间分配
 * @return 实际可用的 maxIndexLen
 */
template <typename T, bool largest>
__aicore__ inline uint32_t RadixTopKWs<T, largest>::CreateVecIndex4TopK()
{
    uint64_t int32Align = BLOCK_SIZE / sizeof(int32_t);
    uint32_t maxIndexLen = Ops::Base::FloorDiv(MAX_TILE_NUM_IN_UB - this->tileNumAlign_, int32Align) * int32Align;
    if (this->tileLen_ < maxIndexLen) {
        maxIndexLen = this->tileLen_;
    } else if (maxIndexLen < 512) {
        maxIndexLen = 0;
    }
    if (maxIndexLen > 0) {
        LocalTensor<int32_t> vecIndex = tileHistAndTopKBuf_.GetWithOffset<int32_t>(
            maxIndexLen, this->tileNumAlign_ * sizeof(int32_t));
        Base::CreateVecIndex4TopK(maxIndexLen, vecIndex);
    }
    return maxIndexLen;
}

/**
 * @brief WS 变体：从 WS 分段读取 tileTopK 查找第一个有 TopK 的 tile
 * @param tileTopK tileTopK UB buffer
 * @return 第一个有 TopK 的 tile ID
 */
template <typename T, bool largest>
__aicore__ inline uint64_t RadixTopKWs<T, largest>::FindFirstTileWithTopK(const LocalTensor<int32_t>& tileTopK)
{
    uint64_t firstId = 0;
    for (int i = 0; i <= tileNumRepeatTimes_; i++) {
        int32_t startTileId = i * MAX_TILE_NUM_IN_UB;
        int32_t id = 0;
        uint32_t dataNum = i == tileNumRepeatTimes_ ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
        MTE2ToSSync();
        while (id < dataNum && tileTopK.GetValue(id) <= 0)
            id++;
        if (id < dataNum) {
            firstId = startTileId + id;
            break;
        }
    }
    return firstId;
}

/**
 * @brief WS 变体 TileTopK：从 WS 分段读取 tileTopK，按 tileTopK 计数提取 TopK 元素并写回输出
 *        按 MAX_TILE_NUM_IN_UB 分段处理，Ping-Pong 双缓冲
 * @param batchId 当前 batch ID
 * @param blockOffset 当前 core 在输入数据中的起始偏移
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::TileTopK(const uint64_t& batchId, const uint64_t& blockOffset)
{
    LocalTensor<int32_t> tmpLocal = tempBuf_.Get<int32_t>();
    uint64_t coreTopKOffset = this->CopyInCoreTopK(tmpLocal);
    uint64_t batchIndexOffset = batchId * this->kValue_;
    coreTopKOffset += batchIndexOffset;

    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    uint64_t firstId = FindFirstTileWithTopK(tileTopK);

    uint64_t firstLen = (firstId == this->tileNum_ - 1) ? this->tailTileLen_ : this->tileLen_;
    this->CopyIn(this->xBufPing_, firstLen, blockOffset + firstId * this->tileLen_);
    uint32_t maxIndexLen = CreateVecIndex4TopK();

    this->InitTopKEvent();
    uint64_t startRepeatId = firstId / MAX_TILE_NUM_IN_UB;
    for (uint64_t repeatId = startRepeatId; repeatId <= tileNumRepeatTimes_; repeatId++) {
        uint64_t startTileId = repeatId * MAX_TILE_NUM_IN_UB;
        uint64_t endTileId = AscendC::Std::min(static_cast<uint64_t>(startTileId + MAX_TILE_NUM_IN_UB),
                                               static_cast<uint64_t>(this->tileNum_));
        if (repeatId > startRepeatId) {
            CopyTileTopKWs2Ub(tileTopK, repeatId * MAX_TILE_NUM_IN_UB, MAX_TILE_NUM_IN_UB);
            MTE2ToSSync();
        } else {
            startTileId = firstId;
        }
        for (uint64_t tileId = startTileId; tileId < endTileId; tileId++) {
            PINGPONG_TILE_BEGIN(tileId, blockOffset);

            uint64_t tileBaseOffset = blockOffset - batchId * this->sortLen_;
            uint64_t tileOffset = tileBaseOffset + tileId * this->tileLen_;

            int32_t curTileK = tileTopK.GetValue(tileId - repeatId * MAX_TILE_NUM_IN_UB);
            PipeBarrier<PIPE_V>();
            PROCESS_ONE_TILE_TOPK_OR_SKIP(curTileK, tileHistAndTopKBuf_.GetWithOffset<int32_t>(
                                                        maxIndexLen, this->tileNumAlign_ * sizeof(int32_t)));

            PINGPONG_TILE_END(tileId);
        }
    }
    this->FinishTopKEvent();
}

} // namespace RadixTopK
#endif // RADIX_TOPK_WS_H
