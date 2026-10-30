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
 * \file radix_top_k_ws_utils.h
 * \brief WS 变体 workspace 数据搬运工具（Clear / Copy / Sub / Add 系列）
 */

#ifndef RADIX_TOPK_WS_UTILS_H
#define RADIX_TOPK_WS_UTILS_H

/**
 * @brief 将workspace的tileTopK 全部清零
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::ClearTileTopKInWs()
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    for (int i = 0; i < tileNumRepeatTimes_; i++) {
        Duplicate<int32_t>(tileTopK, 0, MAX_TILE_NUM_IN_UB);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, i * MAX_TILE_NUM_IN_UB, MAX_TILE_NUM_IN_UB);
        MTE3ToVSync();
    }
    if (tileNumRemain_ > 0) {
        Duplicate<int32_t>(tileTopK, 0, tileNumRemain_);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, tileNumRepeatTimes_ * MAX_TILE_NUM_IN_UB, tileNumRemain_);
    }
}

/**
 * @brief 在主循环结束后的 WS 同步结束阶段：清除 tileHist 中非首 bin 的数据
 * @param dstOffset tileHistGm 目标偏移
 * @param dataNum 清除元素数
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::ClearTileHistInWs(const uint32_t& dstOffset, const uint32_t& dataNum)
{
    if (dataNum == 0)
        return;
    uint32_t repeatTimes = dataNum / MAX_TILE_NUM_IN_UB;
    uint32_t remain = dataNum % MAX_TILE_NUM_IN_UB;
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    for (int i = 0; i < repeatTimes; i++) {
        Duplicate<int32_t>(tileHist, 0, MAX_TILE_NUM_IN_UB);
        VToMTE3Sync();
        DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1),
                                            static_cast<uint32_t>(sizeof(int32_t) * MAX_TILE_NUM_IN_UB), 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + i * MAX_TILE_NUM_IN_UB], tileHist, tileHistOutParams);
        MTE3ToVSync();
    }
    if (remain > 0) {
        Duplicate<int32_t>(tileHist, 0, remain);
        VToMTE3Sync();
        DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * remain),
                                            0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + repeatTimes * MAX_TILE_NUM_IN_UB], tileHist, tileHistOutParams);
    }
}

/**
 * @brief 将 tileHist 从 WS 分段拷入 UB
 * @param dstTensor UB 目标 Tensor
 * @param srcOffset tileHistGm 源偏移
 * @param dataNum 拷贝元素数
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::CopyTileHistWs2Ub(const LocalTensor<int32_t>& dstTensor,
                                                                  const uint32_t& srcOffset, const uint32_t& dataNum)
{
    DataCopyExtParams tileHistInParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0,
                                       0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, tileHistGm_[srcOffset], tileHistInParams, padParams);
}

/**
 * @brief 将 tileTopK 从 WS 分段拷入 UB
 * @param dstTensor UB 目标 Tensor
 * @param srcOffset tileTopKGm 源偏移
 * @param dataNum 拷贝元素数
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::CopyTileTopKWs2Ub(const LocalTensor<int32_t>& dstTensor,
                                                                  const uint32_t& srcOffset, const uint32_t& dataNum)
{
    DataCopyExtParams tileTopKInParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0,
                                       0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, tileTopKGm_[srcOffset], tileTopKInParams, padParams);
}

/**
 * @brief 将 tileTopK 从 UB 拷回 WS
 * @param srcTensor UB 源 Tensor
 * @param dstOffset tileTopKGm 目标偏移
 * @param dataNum 拷贝元素数
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::CopyTileTopKUb2Ws(const LocalTensor<int32_t>& srcTensor,
                                                                  const uint32_t& dstOffset, const uint32_t& dataNum)
{
    DataCopyExtParams tileTopKOutParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0,
                                        0, 0};
    DataCopyPad(tileTopKGm_[dstOffset], srcTensor, tileTopKOutParams);
}

/**
 * @brief 在 WS 中复制 tileHist 整段数据（按 MAX_TILE_NUM_IN_UB 分段）
 * @param dstOffset 目标偏移
 * @param srcOffset 源偏移
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::CopyTileHistInWs(const uint32_t& dstOffset, const uint32_t& srcOffset)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    for (int i = 0; i <= tileNumRepeatTimes_; i++) {
        uint32_t dataNum = i == tileNumRepeatTimes_ ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileHistWs2Ub(tileHist, srcOffset + i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToMTE3Sync();
        DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum),
                                            0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + i * MAX_TILE_NUM_IN_UB], tileHist, tileHistOutParams);
    }
}

/**
 * @brief 从 WS 读入两个 tileHist 段到 UB 并做减法（用于非首轮获取边界 bin 前一个 bin 的数据差）
 * @param dstTensor 减法结果 UB Tensor
 * @param srcTensor 临时 UB Tensor
 * @param srcOffset0 减数源偏移（boundaryBin * tileNum_）
 * @param srcOffset1 被减数源偏移（boundaryBinPrev * tileNum_）
 * @param dataNum 数据量
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::SubTileHistWs2Ub(const LocalTensor<int32_t>& dstTensor,
                                                                 const LocalTensor<int32_t>& srcTensor,
                                                                 const uint32_t& srcOffset0, const uint32_t& srcOffset1,
                                                                 const uint32_t& dataNum)
{
    DataCopyExtParams tileHistInParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0,
                                       0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(srcTensor, tileHistGm_[srcOffset0], tileHistInParams, padParams);
    DataCopyPad(srcTensor[MAX_TILE_NUM_IN_UB_BY2], tileHistGm_[srcOffset1], tileHistInParams, padParams);
    MTE2ToVSync();
    Sub(dstTensor, srcTensor, srcTensor[MAX_TILE_NUM_IN_UB_BY2], dataNum);
}

/**
 * @brief 从 WS 读入两个 tileHist 段到 UB 做减法，再将结果写回 WS
 * @param dstOffset 结果目标偏移
 * @param srcOffset0 减数源偏移
 * @param srcOffset1 被减数源偏移
 * @param dataNum 数据量
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::SubTileHistInWs(const uint32_t& dstOffset, const uint32_t& srcOffset0,
                                                                const uint32_t& srcOffset1, const uint32_t& dataNum)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    SubTileHistWs2Ub(tileHist, tileHist, srcOffset0, srcOffset1, dataNum);
    VToMTE3Sync();
    DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0,
                                        0, 0};
    DataCopyPad(tileHistGm_[dstOffset], tileHist, tileHistOutParams);
}

/**
 * @brief 按 MAX_TILE_NUM_IN_UB_BY2 分片，从 WS 读入两个 tileHist 段并做减法
 * @param dstOffset 结果目标偏移
 * @param srcOffset0 减数源偏移
 * @param srcOffset1 被减数源偏移
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::SubTileHistInWsAll(const uint32_t& dstOffset,
                                                                   const uint32_t& srcOffset0,
                                                                   const uint32_t& srcOffset1)
{
    for (int i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = i == tileNumBy2RepeatTimes_ ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;
        SubTileHistInWs(dstOffset + i * MAX_TILE_NUM_IN_UB_BY2, srcOffset0 + i * MAX_TILE_NUM_IN_UB_BY2,
                        srcOffset1 + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE3ToMTE2Sync();
    }
}

/**
 * @brief WS 变体：从 WS 分段读取 tileTopK，累加边界 bin 之前的 tileHist 并写回 WS
 * @param tileHistOffset tileHist 在 WS 中的偏移（boundaryBin * tileNum_ 或 boundaryBinPrev * tileNum_）
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::AddTileHist2TileTopKInWs(const uint64_t& tileHistOffset)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> tileTopK = tileHist[MAX_TILE_NUM_IN_UB_BY2];
    for (int i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = i == tileNumBy2RepeatTimes_ ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;
        CopyTileHistWs2Ub(tileHist, tileHistOffset + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE2ToVSync();
        Add(tileTopK, tileTopK, tileHist, dataNum);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE3ToMTE2Sync();
    }
}

/**
 * @brief WS 变体：清除/准备当前轮的 tileHist 和 globalHist
 * @param roundId 当前轮 ID
 */
template <typename T, bool largest>
__aicore__ inline void RadixTopKWs<T, largest>::ClearHistInWs(const int32_t& roundId)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = this->globalHistBuf_.template Get<int32_t>();
    MTE3ToVSync();
    if (roundId == this->round_ - 1) {
        for (int i = 0; i < tileNumRepeatTimes_; i++) {
            Duplicate<int32_t>(tileHist, this->tileLen_, MAX_TILE_NUM_IN_UB);
            if (tileNumRemain_ == 0 && i == tileNumRepeatTimes_ - 1 && this->tailTileLen_ != this->tileLen_) {
                VToSSync();
                tileHist.SetValue(MAX_TILE_NUM_IN_UB - 1, this->tailTileLen_);
                SToMTE3Sync();
            } else {
                VToMTE3Sync();
            }
            DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1),
                                                static_cast<uint32_t>(sizeof(int32_t) * MAX_TILE_NUM_IN_UB), 0, 0, 0};
            DataCopyPad(tileHistGm_[i * MAX_TILE_NUM_IN_UB], tileHist, tileHistOutParams);
            MTE3ToVSync();
        }
        if (tileNumRemain_ > 0) {
            Duplicate<int32_t>(tileHist, this->tileLen_, tileNumRemain_);
            if (this->tailTileLen_ != this->tileLen_) {
                VToSSync();
                tileHist.SetValue(tileNumRemain_ - 1, this->tailTileLen_);
                SToMTE3Sync();
            } else {
                VToMTE3Sync();
            }
            DataCopyExtParams tileHistOutParams{static_cast<uint16_t>(1),
                                                static_cast<uint32_t>(sizeof(int32_t) * tileNumRemain_), 0, 0, 0};
            DataCopyPad(tileHistGm_[tileNumRepeatTimes_ * MAX_TILE_NUM_IN_UB], tileHist, tileHistOutParams);
        }
        this->globalHistBoundaryNum_ = (this->tileNum_ - 1) * this->tileLen_ + this->tailTileLen_;
    } else {
        if (this->boundaryBin < this->numValue_ - 1) {
            SubTileHistInWsAll(0, this->boundaryBin * this->tileNum_, this->boundaryBinPrev * this->tileNum_);
        } else {
            CopyTileHistInWs(0, this->boundaryBin * this->tileNum_);
        }
    }
    MTE3ToSSync();
    globalHist.SetValue(0, this->globalHistBoundaryNum_);
    ClearTileHistInWs(this->tileNum_, (this->numValue_ - 1) * this->tileNum_);
}

#endif // RADIX_TOPK_WS_UTILS_H
