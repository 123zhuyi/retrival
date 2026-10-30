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
 * \file chunk_cat.h
 * \brief
 */

#ifndef _CHUNK_CAT_H_
#define _CHUNK_CAT_H_

#include "chunk_cat_common.h"

using namespace AscendC;
template <typename T1, typename T2, bool NEED_CAST = false>
class ChunkCat : public ChunkCatCommon<T1, T2> {
public:
    __aicore__ inline ChunkCat(TPipe* pipe) : ChunkCatCommon<T1, T2>(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ChunkCatTilingData& tilingData)
    {
        this->InitCommon(x, y, tilingData);
        // 获取tiling信息
        isAllAlign_ = tilingData.isAllAlign;
        isHalfAlign_ = tilingData.isHalfAlign;
        isOneConcat_ = tilingData.isOneConcat;
        colRepeatNum_ = isHalfAlign_ ? HALF : this->srcEleUbBlock_;

        srcLocalT2_ = this->srcLocal_.template ReinterpretCast<T2>();
        dstLocalT1_ = this->dstLocal_.template ReinterpretCast<T1>();
        dstLocalFP32_ = this->dstLocal_.template ReinterpretCast<float>();
    }

    __aicore__ inline void Process()
    {
        int64_t rowLoop = this->GetAlign(this->currentBlockRowFactor_, this->ubRowFactor_) / this->ubRowFactor_;
        int64_t colLoop = this->GetAlign(this->currentBlockColFactor_, this->ubColFactor_) / this->ubColFactor_;
        int64_t rowTail = this->currentBlockRowFactor_ % this->ubRowFactor_;
        int64_t colTail = this->currentBlockColFactor_ % this->ubColFactor_;

        uint64_t buf[10];
        this->desc_.SetShapeAddr(buf); // 用于获取shape信息
        int64_t inputCol[32];

        for (int64_t i = 0; i < rowLoop * colLoop; i++) {
            UbLoopInfo ubLoopInfo{};
            ubLoopInfo.inputCol = inputCol;
            ubLoopInfo.ubRowGroup = i / colLoop;
            ubLoopInfo.ubColGroup = i % colLoop;
            ubLoopInfo.currentUbRowFactor = (rowTail != 0 && ubLoopInfo.ubRowGroup == rowLoop - 1) ? rowTail :
                                                                                                     this->ubRowFactor_;
            ubLoopInfo.currentUbColFactor = (colTail != 0 && ubLoopInfo.ubColGroup == colLoop - 1) ? colTail :
                                                                                                     this->ubColFactor_;
            // 1、清零ub
            dupToZero();
            // 搬入
            CopyIn(ubLoopInfo);
            // 计算
            Compute(ubLoopInfo);
            // 搬出
            CopyOut(ubLoopInfo);
        }
    }

private:
    __aicore__ inline void CopyIn(UbLoopInfo& ubLoopInfo)
    {
        // 2、遍历tensor搬运
        int64_t totalCol = 0;
        int64_t localOffset = 0;
        ubLoopInfo.colStart = this->blockColGroup_ * this->blockColFactor_ + ubLoopInfo.ubColGroup * this->ubColFactor_;
        ubLoopInfo.rowStart = this->blockRowGroup_ * this->blockRowFactor_ + ubLoopInfo.ubRowGroup * this->ubRowFactor_;

        for (uint32_t i = 0; i < this->inputNum_; i++) {
            if (ubLoopInfo.totalUbCol >= ubLoopInfo.currentUbColFactor) {
                break;
            }
            this->srcGlobal_.SetGlobalBuffer(this->inputList_.template GetDataPtr<T1>(i));
            TensorInfo tensorInfo{};
            this->GetChunkInfo(i, tensorInfo);
            // 判断当前核是否处理当前tensor
            if (!this->IsTensorInRange(totalCol, ubLoopInfo, tensorInfo)) {
                totalCol += tensorInfo.tensorCol;
            } else {
                this->SplitTensorDim0(totalCol, ubLoopInfo, tensorInfo);
                ubLoopInfo.inputCol[ubLoopInfo.count] = (!isOneConcat_ && !isAllAlign_ && tensorInfo.isSplit) ?
                                                            -tensorInfo.splitCol :
                                                            tensorInfo.splitCol;
                tensorInfo.chunkRow = tensorInfo.chunkDimSize / tensorInfo.chunkCol;
                tensorInfo.chunkRowAlign = this->GetAlign(tensorInfo.chunkDimSize, tensorInfo.chunkCol) /
                                           tensorInfo.chunkCol;
                int64_t localOffsetIncrement = (isOneConcat_ || isAllAlign_) ?
                                                   ubLoopInfo.currentUbRowFactor :
                                                   (isHalfAlign_ ? TRANS_BLOCK * HALF :
                                                                   TRANS_BLOCK * this->srcEleUbBlock_);
                if (ubLoopInfo.rowStart >= tensorInfo.chunkRowAlign) {
                    ubLoopInfo.inputCol[ubLoopInfo.count] = tensorInfo.splitCol;
                    ubLoopInfo.totalUbColAlign += tensorInfo.splitCol;
                    localOffsetIncrement *= tensorInfo.splitCol;
                } else {
                    this->CopyInChunk(totalCol, localOffset, ubLoopInfo, tensorInfo);
                    localOffsetIncrement *= (isOneConcat_ || tensorInfo.isSplit) ? tensorInfo.splitColAlign :
                                                                                   tensorInfo.splitCol;
                    ubLoopInfo.totalUbColAlign += (isOneConcat_ || (!isAllAlign_ && tensorInfo.isSplit)) ?
                                                      tensorInfo.splitColAlign :
                                                      tensorInfo.splitCol;
                }
                ubLoopInfo.totalUbCol += tensorInfo.splitCol;
                ubLoopInfo.count++;
                totalCol += tensorInfo.tensorCol;
                localOffset += localOffsetIncrement;
            }
            ComputeOver32(totalCol, localOffset, ubLoopInfo, tensorInfo);
        }
    }

    __aicore__ inline void Compute(const UbLoopInfo& ubLoopInfo)
    {
        if (isOneConcat_) {
            ComputeOneConcat(ubLoopInfo);
        } else if (ubLoopInfo.isAllZero) {
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE3_V>(this->event_);
            WaitFlag<HardEvent::MTE3_V>(this->event_);
            if constexpr (NEED_CAST) {
                uint32_t castCount = ubLoopInfo.currentUbRowFactor *
                                     this->GetAlign(ubLoopInfo.currentUbColFactor, this->dstEleUbBlock_);
                DoCast(ubLoopInfo, castCount);
            } else {
                DataCopy(dstLocalT1_, this->srcLocal_,
                         ubLoopInfo.currentUbRowFactor *
                             this->GetAlign(ubLoopInfo.currentUbColFactor, this->dstEleUbBlock_));
            }
        } else if (ubLoopInfo.count == 1 && ubLoopInfo.currentUbColFactor % this->srcEleUbBlock_ == 0) {
            SetFlag<HardEvent::MTE2_V>(this->event_);
            WaitFlag<HardEvent::MTE2_V>(this->event_);
            if constexpr (NEED_CAST) {
                uint32_t castCount = ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor;
                DoCast(ubLoopInfo, castCount);
            } else {
                DataCopy(dstLocalT1_, this->srcLocal_, ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor);
            }
        } else if (isAllAlign_) {
            ComputeAllAlign(ubLoopInfo);
        } else {
            ComputeNotAlign(ubLoopInfo);
        }
        SetFlag<HardEvent::V_MTE3>(this->event_);
        WaitFlag<HardEvent::V_MTE3>(this->event_);
    }

    __aicore__ inline void CopyOut(const UbLoopInfo& ubLoopInfo)
    {
        if (isOneConcat_) {
            int64_t localOffset = 0;
            int64_t globalOffset = ubLoopInfo.ubRowGroup * this->ubRowFactor_ * this->outputCol_ +
                                   ubLoopInfo.ubColGroup * this->ubColFactor_ + ubLoopInfo.preCatCol;
            for (int i = 0; i < ubLoopInfo.count; i++) {
                uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
                uint32_t blockLen = ubLoopInfo.inputCol[i] * sizeof(T2);
                uint32_t dstStride = (this->outputCol_ - ubLoopInfo.inputCol[i]) * sizeof(T2);
                DataCopyExtParams copyParamsOut{blockCount, blockLen, 0, dstStride, 0};
                DataCopyPad(this->dstGlobal_[globalOffset], this->dstLocal_[localOffset], copyParamsOut);
                localOffset += this->GetAlign(ubLoopInfo.inputCol[i], this->srcEleUbBlock_);
                globalOffset += ubLoopInfo.inputCol[i];
            }
            SetFlag<HardEvent::MTE3_MTE2>(this->event_);
            WaitFlag<HardEvent::MTE3_MTE2>(this->event_);
            return;
        }
        uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
        uint32_t blockLen = ubLoopInfo.currentUbColFactor * sizeof(T2);
        uint32_t dstStride = (this->outputCol_ - ubLoopInfo.currentUbColFactor) * sizeof(T2);
        DataCopyExtParams copyParamsOut{blockCount, blockLen, 0, dstStride, 0};
        int64_t dstOffset = ubLoopInfo.ubRowGroup * this->ubRowFactor_ * this->outputCol_ +
                            ubLoopInfo.ubColGroup * this->ubColFactor_;
        DataCopyPad(this->dstGlobal_[dstOffset], this->dstLocal_, copyParamsOut);
        SetFlag<HardEvent::MTE3_MTE2>(this->event_);
        WaitFlag<HardEvent::MTE3_MTE2>(this->event_);
    }

    __aicore__ inline void ComputeOver32(int64_t& totalCol, int64_t& localOffset, UbLoopInfo& ubLoopInfo,
                                         TensorInfo& tensorInfo)
    {
        if (isOneConcat_ && ubLoopInfo.count > NUM_THIRTY_ONE) {
            // 计算
            Compute(ubLoopInfo);
            // 搬出
            CopyOut(ubLoopInfo);
            localOffset = 0;
            ubLoopInfo.preCatCol += ubLoopInfo.totalUbCol;
            ubLoopInfo.count = 0;
            ubLoopInfo.totalUbCol = 0;
            ubLoopInfo.totalUbColAlign = 0;
        } else if (ubLoopInfo.count > NUM_THIRTY_ONE) {
            // 提前做部分concat
            if (!ubLoopInfo.isAllZero) {
                SetFlag<HardEvent::MTE2_V>(this->event_);
                WaitFlag<HardEvent::MTE2_V>(this->event_);
                if (isAllAlign_) {
                    UBRearrange4Concat(ubLoopInfo, this->srcLocal_, dstLocalT1_);
                    DataCopy(this->srcLocal_, dstLocalT1_, ubLoopInfo.currentUbRowFactor * ubLoopInfo.totalUbCol);
                } else {
                    // 3、ub重排
                    UBRearrange4Trans(ubLoopInfo, this->srcLocal_, dstLocalT1_);
                    // 4、跨block对齐转置
                    Trans1(ubLoopInfo, dstLocalT1_, this->srcLocal_);
                    // 5、ub重排
                    UBRearrange4TransConcat<true>(ubLoopInfo, this->srcLocal_, dstLocalT1_);
                    // 6、跨block对齐转置
                    Trans2<true>(ubLoopInfo, dstLocalT1_, this->srcLocal_);
                }
                SetFlag<HardEvent::V_MTE2>(this->event_);
                WaitFlag<HardEvent::V_MTE2>(this->event_);
            }
            ubLoopInfo.inputCol[0] = ubLoopInfo.totalUbCol;
            ubLoopInfo.count = 1;
            ubLoopInfo.totalUbColAlign = ubLoopInfo.totalUbCol;
            localOffset = isAllAlign_ ? ubLoopInfo.currentUbRowFactor * ubLoopInfo.totalUbCol :
                                        (isHalfAlign_ ? TRANS_BLOCK * HALF * ubLoopInfo.totalUbCol :
                                                        TRANS_BLOCK * this->srcEleUbBlock_ * ubLoopInfo.totalUbCol);
        }
    }

    __aicore__ inline void dupToZero()
    {
        T1 inputVal(0.0);
        Duplicate<T1>(this->srcLocal_, inputVal, this->srcLocal_.GetSize());
        SetFlag<HardEvent::V_MTE2>(this->event_);
        WaitFlag<HardEvent::V_MTE2>(this->event_);
    }

    __aicore__ inline void UBRearrange4Trans(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal,
                                             LocalTensor<T1>& dstLocal)
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (int64_t i = 0; i < ubLoopInfo.count; i++) {
            uint16_t blockCount = TRANS_BLOCK;
            uint16_t actualCol = ubLoopInfo.inputCol[i] > 0 ?
                                     ubLoopInfo.inputCol[i] :
                                     this->GetAlign(-ubLoopInfo.inputCol[i], this->srcEleUbBlock_);
            uint16_t blockLen = actualCol * colRepeatNum_ / this->srcEleUbBlock_;
            uint16_t dstGap = ubLoopInfo.totalUbColAlign * colRepeatNum_ / this->srcEleUbBlock_ - blockLen;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstGap};
            DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], copyParams);
            srcOffset += blockCount * blockLen * this->srcEleUbBlock_;
            dstOffset += blockLen * this->srcEleUbBlock_;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Trans1(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        uint8_t repeatTimes = ubLoopInfo.totalUbColAlign * colRepeatNum_ / this->srcEleUbBlock_;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};

        uint64_t srcLocalList[TRANS_BLOCK];
        uint64_t dstLocalList[TRANS_BLOCK];
        if constexpr (sizeof(T1) == 2) {
            LocalTensor<half> srcLocalFP16 = srcLocal.template ReinterpretCast<half>();
            LocalTensor<half> dstLocalFP16 = dstLocal.template ReinterpretCast<half>();
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * ubLoopInfo.totalUbColAlign * colRepeatNum_;
                srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocalFP16[offset].GetPhyAddr());
            }
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * TRANS_BLOCK;
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocalFP16[offset].GetPhyAddr());
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        } else {
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * ubLoopInfo.totalUbColAlign * colRepeatNum_;
                srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            for (uint64_t i = 0; i < this->srcEleUbBlock_; i++) {
                for (uint64_t j = 0; j < TRANS_BLOCK / this->srcEleUbBlock_; j++) {
                    uint64_t offset = i * TRANS_BLOCK + j * this->srcEleUbBlock_;
                    dstLocalList[i * TRANS_BLOCK / this->srcEleUbBlock_ + j] = reinterpret_cast<uint64_t>(
                        dstLocal[offset].GetPhyAddr());
                }
            }
            TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool NO_NEED_ALIGN = false>
    __aicore__ inline void UBRearrange4TransConcat(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal,
                                                   LocalTensor<T1>& dstLocal)
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (int64_t i = 0; i < ubLoopInfo.count; i++) {
            uint16_t blockCount = colRepeatNum_;
            uint16_t actualCol = ubLoopInfo.inputCol[i] > 0 ? ubLoopInfo.inputCol[i] : -ubLoopInfo.inputCol[i];
            uint16_t blockLen = actualCol * TRANS_BLOCK / this->srcEleUbBlock_;
            uint16_t srcGap = ubLoopInfo.inputCol[i] > 0 ?
                                  0 :
                                  (this->GetAlign(-ubLoopInfo.inputCol[i], this->srcEleUbBlock_) +
                                   ubLoopInfo.inputCol[i]) *
                                      TRANS_BLOCK / this->srcEleUbBlock_;
            uint16_t dstGap = this->GetAlign(ubLoopInfo.totalUbCol, this->dstEleUbBlock_) * TRANS_BLOCK /
                                  this->srcEleUbBlock_ -
                              blockLen;
            if constexpr (NO_NEED_ALIGN) {
                dstGap = ubLoopInfo.totalUbCol * TRANS_BLOCK / this->srcEleUbBlock_ - blockLen;
            }
            DataCopyParams copyParams{blockCount, blockLen, srcGap, dstGap};
            DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], copyParams);
            srcOffset += ubLoopInfo.inputCol[i] > 0 ?
                             (colRepeatNum_ * actualCol * TRANS_BLOCK) :
                             (colRepeatNum_ * this->GetAlign(actualCol, this->srcEleUbBlock_) * TRANS_BLOCK);
            dstOffset += (actualCol * TRANS_BLOCK);
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool NO_NEED_ALIGN = false>
    __aicore__ inline void Trans2(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        int64_t actualTotalUbCol = this->GetAlign(ubLoopInfo.totalUbCol, this->dstEleUbBlock_);
        if constexpr (NO_NEED_ALIGN) {
            actualTotalUbCol = ubLoopInfo.totalUbCol;
        }
        uint8_t repeatTimes = actualTotalUbCol * colRepeatNum_ / this->srcEleUbBlock_;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : 1;
        TransDataTo5HDParams transDataParams = {false, false, repeatTimes, dstRepStride, srcRepStride};
        uint64_t srcLocalList[TRANS_BLOCK];
        uint64_t dstLocalList[TRANS_BLOCK];
        if (sizeof(T1) == 2) {
            LocalTensor<half> srcLocalFP16 = srcLocal.template ReinterpretCast<half>();
            LocalTensor<half> dstLocalFP16 = dstLocal.template ReinterpretCast<half>();
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * TRANS_BLOCK;
                srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocalFP16[offset].GetPhyAddr());
            }
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * actualTotalUbCol * colRepeatNum_;
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocalFP16[offset].GetPhyAddr());
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        } else {
            for (uint64_t i = 0; i < TRANS_BLOCK / this->srcEleUbBlock_; i++) {
                for (uint64_t j = 0; j < this->srcEleUbBlock_; j++) {
                    uint64_t offset = i * this->srcEleUbBlock_ + j * TRANS_BLOCK;
                    srcLocalList[i * this->srcEleUbBlock_ + j] = reinterpret_cast<uint64_t>(
                        srcLocal[offset].GetPhyAddr());
                }
            }
            for (uint64_t i = 0; i < TRANS_BLOCK; i += 2) {                   // 2 is stride
                uint64_t offset = (i / 2) * actualTotalUbCol * colRepeatNum_; // 2 is stride
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
            for (uint64_t i = 1; i < TRANS_BLOCK; i += 2) {                                          // 2 is stride
                uint64_t offset = (i / 2 + this->srcEleUbBlock_) * actualTotalUbCol * colRepeatNum_; // 2 is stride
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
            TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
        }

        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UBRearrange4Concat(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal,
                                              LocalTensor<T1>& dstLocal)
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (int64_t i = 0; i < ubLoopInfo.count; i++) {
            uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
            uint16_t blockLen = ubLoopInfo.inputCol[i] / this->srcEleUbBlock_;
            uint16_t srcGap = 0;
            uint16_t dstGap = (ubLoopInfo.totalUbColAlign - ubLoopInfo.inputCol[i]) / this->srcEleUbBlock_;
            DataCopyParams copyParams{blockCount, blockLen, srcGap, dstGap};
            DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], copyParams);
            srcOffset += blockCount * ubLoopInfo.inputCol[i];
            dstOffset += ubLoopInfo.inputCol[i];
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoCast(const UbLoopInfo& ubLoopInfo, uint32_t castCount)
    {
        if constexpr (sizeof(T1) == sizeof(T2)) {
            Cast(dstLocalFP32_, this->srcLocal_, RoundMode::CAST_NONE, castCount);
            Cast(srcLocalT2_, dstLocalFP32_, RoundMode::CAST_RINT, castCount);
            DataCopy(this->dstLocal_, srcLocalT2_, castCount);
        } else {
            Cast(this->dstLocal_, this->srcLocal_, RoundMode::CAST_NONE, castCount);
        }
    }

    __aicore__ inline void ComputeOneConcat(const UbLoopInfo& ubLoopInfo)
    {
        SetFlag<HardEvent::MTE2_V>(this->event_);
        WaitFlag<HardEvent::MTE2_V>(this->event_);
        if (ubLoopInfo.totalUbColAlign == 0) {
            return;
        }
        if constexpr (NEED_CAST) {
            uint32_t castCount = ubLoopInfo.currentUbRowFactor * ubLoopInfo.totalUbColAlign;
            DoCast(ubLoopInfo, castCount);
        } else {
            DataCopy(dstLocalT1_, this->srcLocal_, ubLoopInfo.totalUbColAlign);
        }
    }

    __aicore__ inline void ComputeAllAlign(const UbLoopInfo& ubLoopInfo)
    {
        SetFlag<HardEvent::MTE2_V>(this->event_);
        WaitFlag<HardEvent::MTE2_V>(this->event_);
        UBRearrange4Concat(ubLoopInfo, this->srcLocal_, dstLocalT1_);
        if constexpr (NEED_CAST) {
            DataCopy(this->srcLocal_, dstLocalT1_, ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor);
            PipeBarrier<PIPE_V>();
            uint32_t castCount = ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor;
            DoCast(ubLoopInfo, castCount);
        }
    }

    __aicore__ inline void ComputeNotAlign(const UbLoopInfo& ubLoopInfo)
    {
        SetFlag<HardEvent::MTE2_V>(this->event_);
        WaitFlag<HardEvent::MTE2_V>(this->event_);
        // 3、ub重排
        UBRearrange4Trans(ubLoopInfo, this->srcLocal_, dstLocalT1_);
        // 4、跨block对齐转置
        Trans1(ubLoopInfo, dstLocalT1_, this->srcLocal_);
        // 5、ub重排
        UBRearrange4TransConcat(ubLoopInfo, this->srcLocal_, dstLocalT1_);
        // 6、跨block对齐转置
        Trans2(ubLoopInfo, dstLocalT1_, this->srcLocal_);

        // 7、cast or ubToub
        if constexpr (NEED_CAST) {
            uint32_t castCount = ubLoopInfo.currentUbRowFactor *
                                 this->GetAlign(ubLoopInfo.currentUbColFactor, this->dstEleUbBlock_);
            DoCast(ubLoopInfo, castCount);
        } else {
            DataCopy(
                dstLocalT1_, this->srcLocal_,
                ubLoopInfo.currentUbRowFactor * this->GetAlign(ubLoopInfo.currentUbColFactor, this->dstEleUbBlock_));
        }
    }

private:
    bool isAllAlign_{false};
    bool isHalfAlign_{false};
    bool isOneConcat_{false};
    int64_t colRepeatNum_{0};

    LocalTensor<T2> srcLocalT2_;
    LocalTensor<T1> dstLocalT1_;
    LocalTensor<float> dstLocalFP32_;
};
#endif // _CHUNK_CAT_H_
