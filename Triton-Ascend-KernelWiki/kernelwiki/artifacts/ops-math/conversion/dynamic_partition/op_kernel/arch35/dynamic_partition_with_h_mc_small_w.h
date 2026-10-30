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
 * \file dynamic_partition_with_h_mc_small_w.h
 * \brief impl of DynamicPartition which multiple cores at axis H with small w
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_WITH_H_MC_SMALL_W_H_
#define OP_KERNEL_DYNAMIC_PARTITION_WITH_H_MC_SMALL_W_H_

#include "dynamic_partition_with_h_mc.h"

namespace DynPart {
using namespace AscendC;

template <typename T>
class DynPartWithHMCSMALLW : public DynPartWithHMC<T> {
public:
    __aicore__ inline DynPartWithHMCSMALLW(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape, GM_ADDR workspace,
                                const DynPartTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void MultiplePartProcess(TQue<QuePosition::VECCALC, 1>& pBaseQue, int32_t begPart,
                                               int32_t endPart);
    __aicore__ inline void InitProcessBuffer();
    __aicore__ inline void CopyPInBrc(int64_t hLpIdx, uint32_t hLen, uint32_t wLen);
    __aicore__ inline void CopyXPIn(int64_t hLpIdx, uint32_t hLen);
    __aicore__ inline void MultiplePartGatherOutX(LocalTensor<uint64_t>& ubPartBase, uint32_t partLen, int32_t begPart,
                                                  int32_t endPart);
    __aicore__ inline void GatherB32B64(uint32_t processNum, uint16_t loopCnt, uint32_t loopTailNum,
                                        __local_mem__ int32_t* ptrPartMid, __local_mem__ T* ptrXIn,
                                        __local_mem__ T* ptrXOut);
    __aicore__ inline void GatherB8B16(uint32_t processNum, uint16_t loopCnt, uint32_t loopTailNum,
                                       __local_mem__ int32_t* ptrPartMid, __local_mem__ T* ptrXIn,
                                       __local_mem__ T* ptrXOut);

private:
    const DynPartTilingData* tdPtr_ = nullptr;
    int32_t begPart_ = 0;
    int32_t endPart_ = 0;
};

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                                     GM_ADDR workspace, const DynPartTilingData* tilingDataPtr,
                                                     TPipe* pipeIn)
{
    tdPtr_ = tilingDataPtr;
    DynPartWithHMC<T>::Init(x, partitions, y, yshape, workspace, tilingDataPtr, pipeIn);
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::Process()
{
    int32_t partLpCnt = tdPtr_->numPartitions / NUM_PARTITION_UNIT;
    int32_t partLeft = tdPtr_->numPartitions % NUM_PARTITION_UNIT;
    for (int32_t pLpIdx = 0; pLpIdx < partLpCnt; ++pLpIdx) {
        begPart_ = pLpIdx * NUM_PARTITION_UNIT;
        endPart_ = (pLpIdx + 1) * NUM_PARTITION_UNIT;
        this->CalcPartBase(begPart_, endPart_);
        MultiplePartProcess(this->pBaseQue_, begPart_, endPart_);
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, NUM_PARTITION_UNIT,
                                  static_cast<int64_t>(pLpIdx * NUM_PARTITION_UNIT * SHAPE_GAP));
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
    if (partLeft > 0) {
        begPart_ = partLpCnt * NUM_PARTITION_UNIT;
        endPart_ = tdPtr_->numPartitions;
        this->CalcPartBase(begPart_, endPart_);
        MultiplePartProcess(this->pBaseQue_, begPart_, endPart_);
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, partLeft,
                                  static_cast<int64_t>(partLpCnt * NUM_PARTITION_UNIT * SHAPE_GAP));
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::MultiplePartProcess(TQue<QuePosition::VECCALC, 1>& pBaseQue,
                                                                    int32_t begPart, int32_t endPart)
{
    InitProcessBuffer();

    int64_t hSize = (this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->hMSize : tdPtr_->hTSize;
    int64_t hLpCnt = hSize / tdPtr_->hLpUnit;
    uint32_t hLeft = static_cast<uint32_t>(hSize % tdPtr_->hLpUnit);
    for (int64_t hLpIdx = 0; hLpIdx < hLpCnt; ++hLpIdx) {
        CopyXPIn(hLpIdx, static_cast<uint32_t>(tdPtr_->hLpUnit));
        auto ubPartBase = pBaseQue.DeQue<uint64_t>();
        MultiplePartGatherOutX(ubPartBase, tdPtr_->hLpUnit, begPart, endPart);
        pBaseQue.EnQue(ubPartBase);
    }
    if (hLeft > 0U) {
        CopyXPIn(hLpCnt, hLeft);
        auto ubPartBase = pBaseQue.DeQue<uint64_t>();
        MultiplePartGatherOutX(ubPartBase, hLeft, begPart, endPart);
        pBaseQue.EnQue(ubPartBase);
    }
    this->bufPoolProcess_.Reset();
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::InitProcessBuffer()
{
    uint32_t xUBSize = 1;
    xUBSize = static_cast<uint32_t>(
        Ops::Base::CeilAlign(tdPtr_->hLpUnit * tdPtr_->wLpUnit, int64_t(this->elePerBlock_)) * sizeof(T));
    this->bufPoolProcess_.InitBuffer(this->xInQue_, NUM_TWO, xUBSize);
    this->bufPoolProcess_.InitBuffer(this->pR2InQue_, NUM_TWO, tdPtr_->hLpUnit * tdPtr_->wLpUnit * sizeof(int32_t));
    this->bufPoolProcess_.InitBuffer(this->pR2MidBuf_, tdPtr_->hLpUnit * tdPtr_->wLpUnit * sizeof(int32_t));
    this->bufPoolProcess_.InitBuffer(this->xOutQue_, NUM_TWO, xUBSize);
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::CopyPInBrc(int64_t hLpIdx, uint32_t hLen, uint32_t wLen)
{
    auto ubR2PIn = this->pR2InQue_.template AllocTensor<int32_t>();
    constexpr uint8_t dim = NUM_TWO;
    static constexpr MultiCopyConfig config = {false};
    MultiCopyLoopInfo<dim> loopInfo = {.loopSrcStride = {0, 1},
                                       .loopDstStride = {1, wLen},
                                       .loopSize = {wLen, hLen},
                                       .loopLpSize = {0, 0},
                                       .loopRpSize = {0, 0}};
    MultiCopyParams<int32_t, dim> copyInParams = {loopInfo, 0};
    DataCopy<int32_t, dim, config>(ubR2PIn, this->pInGM_[this->blockIdx_ * tdPtr_->hMSize + hLpIdx * tdPtr_->hLpUnit],
                                   copyInParams);
    this->pR2InQue_.EnQue(ubR2PIn);
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::CopyXPIn(int64_t hLpIdx, uint32_t hLen)
{
    uint32_t wLen = static_cast<uint32_t>((this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->wMSize :
                                                                                         tdPtr_->wTSize);
    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(wLen * hLen * sizeof(T)), 0, 0, 0};
    auto ubXIn = this->xInQue_.template AllocTensor<T>();
    DataCopyPad(
        ubXIn,
        this->xInGM_[this->blockIdx_ * tdPtr_->hMSize * tdPtr_->hOffset + hLpIdx * tdPtr_->hLpUnit * tdPtr_->hOffset],
        copyParams, copyPadParams);
    this->xInQue_.EnQue(ubXIn);

    CopyPInBrc(hLpIdx, hLen, wLen);
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::MultiplePartGatherOutX(LocalTensor<uint64_t>& ubPartBase,
                                                                       uint32_t partLen, int32_t begPart,
                                                                       int32_t endPart)
{
    uint32_t wLen = static_cast<uint32_t>((this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->wMSize :
                                                                                         tdPtr_->wTSize);
    auto ubPartMid = this->pR2MidBuf_.template Get<int32_t>();
    auto ubXIn = this->xInQue_.template DeQue<T>();
    auto ubPartIn = this->pR2InQue_.template DeQue<int32_t>();
    auto ubXOut = this->xOutQue_.template AllocTensor<T>();
    __local_mem__ T* ptrXOut = (__local_mem__ T*)ubXOut.GetPhyAddr();
    __local_mem__ T* ptrXIn = (__local_mem__ T*)ubXIn.GetPhyAddr();
    uint32_t ubOffset = 0;
    uint16_t partLpCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(partLen * wLen, this->b32VLSize_));
    int32_t int32VL = static_cast<int32_t>(this->b32VLSize_);
    for (int32_t partID = begPart; partID < endPart; ++partID) {
        __local_mem__ int32_t* ptrPartIn = (__local_mem__ int32_t*)ubPartIn.GetPhyAddr();
        __local_mem__ int32_t* ptrPartMid = (__local_mem__ int32_t*)ubPartMid.GetPhyAddr();
        uint32_t partSize = partLen * wLen;
        __VEC_SCOPE__
        {
            RegTensor<int32_t> partIn;
            RegTensor<int32_t> partMid;
            RegTensor<int32_t> alphaIdx;
            Reg::Arange(alphaIdx, int32_t(0));
            Reg::UnalignReg ureg;
            MaskReg validMask;
            MaskReg cmpMask;
            Reg::ClearSpr<SpecialPurposeReg::AR>();
            for (uint16_t partLpIdx = 0; partLpIdx < partLpCnt; ++partLpIdx) {
                validMask = UpdateMask<int32_t>(partSize);
                Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(partIn, ptrPartIn, int32VL);
                Reg::CompareScalar(cmpMask, partIn, partID, validMask);
                Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(partMid, alphaIdx, cmpMask);
                Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(ptrPartMid, partMid, ureg);
                Reg::Adds(alphaIdx, alphaIdx, int32VL, validMask);
            }
            Reg::DataCopyUnAlignPost(ptrPartMid, ureg);
        }

        uint32_t vPartCnt = static_cast<uint32_t>(Reg::GetSpr<SpecialPurposeReg::AR>() / sizeof(int32_t));
        if (vPartCnt > 0) {
            uint32_t processNum = (sizeof(T) == sizeof(int64_t)) ? int32VL / NUM_TWO : int32VL;
            uint16_t loopCnt = vPartCnt / processNum;
            uint32_t loopTailNum = vPartCnt - loopCnt * processNum;
            if constexpr (sizeof(T) == sizeof(int32_t) || sizeof(T) == sizeof(int64_t)) {
                // 针对x为B32和B64数据类型，index数据类型为uint32直接进行gather处理
                GatherB32B64(processNum, loopCnt, loopTailNum, ptrPartMid, ptrXIn, ptrXOut);
            } else {
                // 针对x为B8和B16数据类型，index数据类型需要pack为uint32进行gather处理。x为B8时，gather结束后还需要pack到B8类型。
                GatherB8B16(processNum, loopCnt, loopTailNum, ptrPartMid, ptrXIn, ptrXOut);
            }
            // to avoid data conflict
            this->InsertSync(HardEvent::V_MTE3);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(vPartCnt * sizeof(T)), 0, 0, 0};
            int32_t modPartID = partID % NUM_PARTITION_UNIT;
            int64_t baseOffset = ubPartBase.GetValue(modPartID);
            this->xOutGM_.SetGlobalBuffer(this->outGMList_.template GetDataPtr<T>(partID));
            int64_t gmOutBaseOffset = baseOffset * tdPtr_->hOffset;
            DataCopyPad(this->xOutGM_[gmOutBaseOffset], ubXOut, copyParams);
            ubPartBase.SetValue(modPartID, baseOffset + vPartCnt / wLen);
        }
    }
    this->xOutQue_.FreeTensor(ubXOut);
    this->xInQue_.FreeTensor(ubXIn);
    this->pR2InQue_.FreeTensor(ubPartIn);
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::GatherB32B64(uint32_t processNum, uint16_t loopCnt,
                                                             uint32_t loopTailNum, __local_mem__ int32_t* ptrPartMid,
                                                             __local_mem__ T* ptrXIn, __local_mem__ T* ptrXOut)
{
    __VEC_SCOPE__
    {
        RegTensor<int32_t> indexReg;
        RegTensor<T> dstReg;
        MaskReg mask;
        for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            uint32_t maskNum = processNum;
            mask = UpdateMask<T>(maskNum);
            Reg::LoadAlign(indexReg, ptrPartMid + loopIdx * processNum);
            Reg::Gather(dstReg, ptrXIn, (AscendC::Reg::RegTensor<uint32_t>&)indexReg, mask);
            Reg::StoreAlign(ptrXOut + loopIdx * processNum, dstReg, mask);
        }
        if (loopTailNum != 0) {
            uint32_t maskNum = loopTailNum;
            mask = UpdateMask<T>(maskNum);
            Reg::LoadAlign(indexReg, ptrPartMid + loopCnt * processNum);
            Reg::Gather(dstReg, ptrXIn, (AscendC::Reg::RegTensor<uint32_t>&)indexReg, mask);
            Reg::StoreAlign(ptrXOut + loopCnt * processNum, dstReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DynPartWithHMCSMALLW<T>::GatherB8B16(uint32_t processNum, uint16_t loopCnt, uint32_t loopTailNum,
                                                            __local_mem__ int32_t* ptrPartMid, __local_mem__ T* ptrXIn,
                                                            __local_mem__ T* ptrXOut)
{
    __VEC_SCOPE__
    {
        RegTensor<int32_t> indexRegB32;
        RegTensor<uint16_t> indexRegB16;
        RegTensor<uint16_t> dstReg;
        RegTensor<T> dstRegT;

        MaskReg mask;
        MaskReg maskB8;
        for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            uint32_t maskNum = processNum;
            uint32_t maskNumB8 = processNum;
            mask = UpdateMask<uint16_t>(maskNum);
            Reg::LoadAlign(indexRegB32, ptrPartMid + loopIdx * processNum);
            Reg::Pack(indexRegB16, indexRegB32);
            Reg::Gather(dstReg, ptrXIn, indexRegB16, mask);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                // Convert B16 to B8
                maskB8 = UpdateMask<T>(maskNumB8);
                Reg::Pack(dstRegT, dstReg);
                Reg::StoreAlign(ptrXOut + loopIdx * processNum, dstRegT, maskB8);
            } else {
                Reg::StoreAlign(ptrXOut + loopIdx * processNum, dstReg, mask);
            }
        }
        if (loopTailNum != 0) {
            uint32_t maskNum = loopTailNum;
            uint32_t maskNumB8 = loopTailNum;
            mask = UpdateMask<uint16_t>(maskNum);
            Reg::LoadAlign(indexRegB32, ptrPartMid + loopCnt * processNum);
            Reg::Pack(indexRegB16, indexRegB32);
            Reg::Gather(dstReg, ptrXIn, indexRegB16, mask);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                // Convert B16 to B8
                maskB8 = UpdateMask<T>(maskNumB8);
                Reg::Pack(dstRegT, dstReg);
                Reg::StoreAlign(ptrXOut + loopCnt * processNum, dstRegT, maskB8);
            } else {
                Reg::StoreAlign(ptrXOut + loopCnt * processNum, dstReg, mask);
            }
        }
    }
}

} // namespace DynPart

#endif // OP_KERNEL_DYNAMIC_PARTITION_WITH_H_MC_SMALL_W_H_
