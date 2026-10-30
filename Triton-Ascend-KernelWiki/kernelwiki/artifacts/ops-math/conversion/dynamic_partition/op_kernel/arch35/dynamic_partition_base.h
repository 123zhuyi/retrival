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
 * \file dynamic_partition_base.h
 * \brief base class of dynamic_partition
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_BASE_H_
#define OP_KERNEL_DYNAMIC_PARTITION_BASE_H_

#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "dynamic_partition_tiling_data_struct.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace DynPart {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t SHAPE_GAP = 9; // dim size + 8 dims
constexpr uint32_t MAX_BUFFER_NUM = 8;

template <typename T>
class DynPartBase {
public:
    __aicore__ inline DynPartBase(){};

protected:
    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                    const DynPartTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void RefreshOutputShapes(LocalTensor<uint64_t>& ubPartBase, int32_t partLen, int64_t offset);
    __aicore__ inline void SinglePartProcess(TQue<QuePosition::VECCALC, 1>& pBaseQue, int32_t begPart, int32_t endPart);
    __aicore__ inline void MultiplePartProcess(TQue<QuePosition::VECCALC, 1>& pBaseQue, int32_t begPart,
                                               int32_t endPart);
    __aicore__ inline void SetPartBaseZero(TQue<QuePosition::VECCALC, 1>& pBaseQue);
    __aicore__ inline void InsertSync(const HardEvent& event);

private:
    __aicore__ inline void MultiplePartCopyOutX(LocalTensor<uint64_t>& ubPartBase, uint32_t partLen, int32_t begPart,
                                                int32_t endPart);
    __aicore__ inline void SinglePartCopyOutX(LocalTensor<uint64_t>& ubPartBase, uint32_t partLen, int32_t begPart,
                                              int32_t endPart, int64_t gmInBaseOffset);
    __aicore__ inline int64_t CalcBlockInOffset();
    __aicore__ inline int64_t CalcBlockOutOffset();
    __aicore__ inline void CopyPIn(int64_t hLpIdx, uint32_t hLen);
    __aicore__ inline void CopyXPIn(int64_t hLpIdx, uint32_t hLen);
    __aicore__ inline void InitProcessBuffer();
    __aicore__ inline void InitAfterProcessBuffer();

protected:
    const DynPartTilingData* tdPtr_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t gmXInBlockOffset_ = 0;
    uint32_t vlSize_ = static_cast<uint32_t>(Ops::Base::GetVRegSize() / sizeof(T));
    uint32_t elePerBlock_ = static_cast<uint32_t>(BLOCK_SIZE / sizeof(T));
    int32_t b64PerBlock = static_cast<int32_t>(BLOCK_SIZE / sizeof(int64_t));

    GlobalTensor<T> xInGM_;
    TQue<QuePosition::VECIN, 1> xInQue_;

    GlobalTensor<int32_t> pInGM_;
    TQue<QuePosition::VECIN, 1> pR2InQue_;
    TBuf<QuePosition::VECCALC> pR2MidBuf_;

    ListTensorDesc outGMList_;
    GlobalTensor<T> xOutGM_;
    TQue<QuePosition::VECOUT, 1> xOutQue_;

    GlobalTensor<uint64_t> yShapeGM_;
    TBuf<QuePosition::VECCALC> shapeBuf_;
    int32_t shapeLpUnit_ = 0;

    TBufPool<QuePosition::VECCALC, MAX_BUFFER_NUM> bufPool_;
    TBufPool<QuePosition::VECCALC, MAX_BUFFER_NUM> bufPoolAfter_;
    TBufPool<QuePosition::VECCALC, MAX_BUFFER_NUM> bufPoolProcess_;
    bool refreshFirstShape_ = true;
    int32_t coreWS_ = 0;
    int32_t coreWSAlign_ = 0;
    int64_t blockIdx_ = 0;
    uint32_t b32VLSize_ = static_cast<uint32_t>(Ops::Base::GetVRegSize() / sizeof(int32_t));
    TBufPool<QuePosition::VECCALC, MAX_BUFFER_NUM> bufPoolSync_;
    TQue<QuePosition::VECCALC, 1> pCntQue_;
    TQue<QuePosition::VECCALC, 1> pBaseQue_;
};

template <typename T>
__aicore__ inline void DynPartBase<T>::BaseInit(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                                const DynPartTilingData* tilingDataPtr, TPipe* pipeIn)
{
    tdPtr_ = tilingDataPtr;
    pipe_ = pipeIn;
    blockIdx_ = AscendC::GetBlockIdx();
    gmXInBlockOffset_ = CalcBlockInOffset();

    coreWS_ = (tdPtr_->numPartitions > NUM_PARTITION_UNIT) ? NUM_PARTITION_UNIT : tdPtr_->numPartitions;
    coreWSAlign_ = Ops::Base::CeilAlign(coreWS_, b64PerBlock);
    uint32_t wsUBSize = static_cast<uint32_t>(coreWSAlign_ * sizeof(uint64_t));
    pipe_->InitBufPool(bufPool_, tdPtr_->totalUBSize);
    bufPool_.InitBuffer(pCntQue_, 1, wsUBSize);
    bufPool_.InitBuffer(pBaseQue_, 1, wsUBSize);
    uint32_t leftUBSize = static_cast<uint32_t>(tdPtr_->totalUBSize - NUM_TWO * wsUBSize);
    bufPool_.InitBufPool(bufPoolProcess_, leftUBSize);
    bufPool_.InitBufPool(bufPoolAfter_, leftUBSize, bufPoolProcess_);
    bufPool_.InitBufPool(bufPoolSync_, leftUBSize, bufPoolProcess_);

    xInGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    pInGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(partitions));
    outGMList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));
    yShapeGM_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(yshape));
}

template <typename T>
__aicore__ inline void DynPartBase<T>::InitProcessBuffer()
{
    uint32_t xUBSize = 1;
    if (tdPtr_->tilingKey == TILING_H_MC_UB_CAN_HOLD_SPLIT_W || tdPtr_->tilingKey == TILING_W_MC_UB_CAN_HOLD_SPLIT_W) {
        xUBSize = static_cast<uint32_t>(tdPtr_->hLpUnit * Ops::Base::CeilAlign(tdPtr_->wLpUnit, int64_t(elePerBlock_)) *
                                        sizeof(T));
    } else if (tdPtr_->tilingKey == TILING_H_MC_UB_CANNOT_HOLD_SPLIT_W ||
               tdPtr_->tilingKey == TILING_W_MC_UB_CANNOT_HOLD_SPLIT_W || tdPtr_->tilingKey == TILING_XP_SCALAR) {
        xUBSize = static_cast<uint32_t>(Ops::Base::CeilAlign(tdPtr_->wLpUnit, int64_t(elePerBlock_)) * sizeof(T));
    }
    bufPoolProcess_.InitBuffer(xInQue_, NUM_TWO, xUBSize);
    bufPoolProcess_.InitBuffer(pR2InQue_, NUM_TWO, tdPtr_->hLpUnit * sizeof(int32_t));
    bufPoolProcess_.InitBuffer(pR2MidBuf_, tdPtr_->hLpUnit * sizeof(int32_t));
    bufPoolProcess_.InitBuffer(xOutQue_, NUM_TWO, xUBSize);
}

template <typename T>
__aicore__ inline void DynPartBase<T>::InitAfterProcessBuffer()
{
    constexpr int32_t halfPartUnit = static_cast<int32_t>(NUM_PARTITION_UNIT / NUM_TWO);
    shapeLpUnit_ = (tdPtr_->numPartitions > halfPartUnit) ? halfPartUnit : tdPtr_->numPartitions;
    bufPoolAfter_.InitBuffer(shapeBuf_, shapeLpUnit_ * SHAPE_GAP * sizeof(uint64_t));
}

template <typename T>
__aicore__ inline void DynPartBase<T>::SetPartBaseZero(TQue<QuePosition::VECCALC, 1>& pBaseQue)
{
    auto ubPartBase = pBaseQue.AllocTensor<uint64_t>();
    Duplicate(ubPartBase, 0UL, coreWSAlign_);
    pBaseQue.EnQue(ubPartBase);
}

template <typename T>
__aicore__ inline void DynPartBase<T>::RefreshOutputShapes(LocalTensor<uint64_t>& ubPartBase, int32_t partLen,
                                                           int64_t offset)
{
    InitAfterProcessBuffer();
    if (blockIdx_ != tdPtr_->usedCoreCnt - 1) {
        return;
    }

    auto ubOutShape = shapeBuf_.Get<uint64_t>();
    Duplicate(ubOutShape, 1UL, static_cast<int32_t>(shapeLpUnit_ * SHAPE_GAP));
    InsertSync(HardEvent::V_S);
    // set first output dimNum and dims exclude dim0
    uint64_t dimNum = static_cast<uint64_t>(tdPtr_->dimNumExtFirst + 1);
    ubOutShape.SetValue(0, dimNum);
    for (uint32_t i = 0; i < static_cast<uint32_t>(tdPtr_->dimNumExtFirst); ++i) {
        ubOutShape.SetValue(i + NUM_TWO, static_cast<uint64_t>(tdPtr_->outDimsExtFirst[i]));
    }

    uint16_t lpCnt = static_cast<uint16_t>(shapeLpUnit_ - 1);
    if (lpCnt > 0) {
        InsertSync(HardEvent::S_V);
        __local_mem__ uint64_t* ptrFirstShape = (__local_mem__ uint64_t*)ubOutShape.GetPhyAddr();
        __local_mem__ uint64_t* ptrSecondShape = ptrFirstShape + SHAPE_GAP;
        __VEC_SCOPE__
        {
            RegTensor<uint64_t> shapeInfo;
            Reg::UnalignReg ureg;
            Reg::DataCopy(shapeInfo, ptrFirstShape);
            for (uint16_t j = 0; j < lpCnt; ++j) {
                Reg::DataCopyUnAlign(ptrSecondShape, shapeInfo, ureg, SHAPE_GAP);
            }
            Reg::DataCopyUnAlignPost(ptrSecondShape, ureg);
        }
        InsertSync(HardEvent::V_S);
    }

    // to mask the data type of shape is uint64
    if (refreshFirstShape_) {
        uint64_t b64Flag = (0x1UL << 31);
        ubOutShape.SetValue(0, dimNum + b64Flag);
    }

    int32_t pLpCnt = partLen / shapeLpUnit_;
    int32_t pLeft = partLen % shapeLpUnit_;
    uint32_t outLen = static_cast<uint32_t>(shapeLpUnit_ * SHAPE_GAP);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(outLen * sizeof(uint64_t)), 0, 0, 0};
    for (int32_t i = 0; i < pLpCnt; ++i) {
        for (int32_t j = 0; j < shapeLpUnit_; ++j) {
            auto dim0 = ubPartBase.GetValue(i * shapeLpUnit_ + j);
            ubOutShape.SetValue(j * SHAPE_GAP + 1, dim0);
        }
        InsertSync(HardEvent::S_MTE3);
        DataCopyPad(yShapeGM_[static_cast<int64_t>(outLen) * i + offset], ubOutShape, copyParams);
        InsertSync(HardEvent::MTE3_S);
        if (refreshFirstShape_) {
            ubOutShape.SetValue(0, dimNum);
            refreshFirstShape_ = false;
        }
    }
    if (pLeft > 0) {
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(pLeft * SHAPE_GAP * sizeof(uint64_t)), 0, 0, 0};
        for (int32_t j = 0; j < pLeft; ++j) {
            auto dim0 = ubPartBase.GetValue(pLpCnt * shapeLpUnit_ + j);
            ubOutShape.SetValue(j * SHAPE_GAP + 1, dim0);
        }
        InsertSync(HardEvent::S_MTE3);
        DataCopyPad(yShapeGM_[static_cast<int64_t>(outLen) * pLpCnt + offset], ubOutShape, copyParams1);
    }
    bufPoolAfter_.Reset();
}

template <typename T>
__aicore__ inline int64_t DynPartBase<T>::CalcBlockInOffset()
{
    if (tdPtr_->tilingKey == TILING_H_MC_UB_CAN_HOLD_SPLIT_W ||
        tdPtr_->tilingKey == TILING_H_MC_UB_CANNOT_HOLD_SPLIT_W) {
        return blockIdx_ * tdPtr_->hMSize * tdPtr_->hOffset;
    }
    if (tdPtr_->tilingKey == TILING_W_MC_UB_CAN_HOLD_SPLIT_W ||
        tdPtr_->tilingKey == TILING_W_MC_UB_CANNOT_HOLD_SPLIT_W) {
        return blockIdx_ * tdPtr_->wMSize;
    }
    return 0L;
}

template <typename T>
__aicore__ inline int64_t DynPartBase<T>::CalcBlockOutOffset()
{
    if (tdPtr_->tilingKey == TILING_W_MC_UB_CAN_HOLD_SPLIT_W ||
        tdPtr_->tilingKey == TILING_W_MC_UB_CANNOT_HOLD_SPLIT_W) {
        return blockIdx_ * tdPtr_->wMSize;
    }
    return 0L;
}

/********************************************************
 * partLen is equal to hLen of x
 ********************************************************/
template <typename T>
__aicore__ inline void DynPartBase<T>::MultiplePartCopyOutX(LocalTensor<uint64_t>& ubPartBase, uint32_t partLen,
                                                            int32_t begPart, int32_t endPart)
{
    auto ubPartMid = pR2MidBuf_.Get<int32_t>();
    auto ubXIn = xInQue_.DeQue<T>();
    auto ubPartIn = pR2InQue_.DeQue<int32_t>();
    auto ubXOut = xOutQue_.AllocTensor<T>();
    __local_mem__ T* ptrXOut = (__local_mem__ T*)ubXOut.GetPhyAddr();
    __local_mem__ T* ptrXIn = (__local_mem__ T*)ubXIn.GetPhyAddr();
    uint32_t ubOffset = 0;
    uint16_t partLpCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(partLen, b32VLSize_));
    int32_t int32VL = static_cast<int32_t>(b32VLSize_);
    for (int32_t partID = begPart; partID < endPart; ++partID) {
        __local_mem__ int32_t* ptrPartIn = (__local_mem__ int32_t*)ubPartIn.GetPhyAddr();
        __local_mem__ int32_t* ptrPartMid = (__local_mem__ int32_t*)ubPartMid.GetPhyAddr();
        uint32_t partSize = partLen;
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
            uint32_t wLen = static_cast<uint32_t>((blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->wMSize :
                                                                                           tdPtr_->wTSize);
            uint32_t wAlign = Ops::Base::CeilAlign(wLen, elePerBlock_);
            uint16_t wLpCnt = Ops::Base::CeilDiv(wLen, vlSize_);
            for (uint32_t vPIdx = 0; vPIdx < vPartCnt; ++vPIdx) {
                uint32_t curWIdx = static_cast<uint32_t>(ubPartMid.GetValue(vPIdx));
                uint32_t wSize = wLen;
                __VEC_SCOPE__
                {
                    RegTensor<T> xReg;
                    MaskReg mask;
                    for (uint16_t wLpIdx = 0; wLpIdx < wLpCnt; ++wLpIdx) {
                        mask = UpdateMask<T>(wSize);
                        Reg::DataCopy(xReg, ptrXIn + curWIdx * wAlign + wLpIdx * vlSize_);
                        Reg::DataCopy(ptrXOut + vPIdx * wAlign + wLpIdx * vlSize_, xReg, mask);
                    }
                }
            }
            // to avoid data conflict
            ptrXOut += vPartCnt * wAlign;

            InsertSync(HardEvent::V_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(vPartCnt), static_cast<uint32_t>(wLen * sizeof(T)), 0,
                                         static_cast<int64_t>((tdPtr_->hOffset - wLen) * sizeof(T)), 0};
            int32_t modPartID = partID % NUM_PARTITION_UNIT;
            int64_t baseOffset = ubPartBase.GetValue(modPartID);
            xOutGM_.SetGlobalBuffer(outGMList_.GetDataPtr<T>(partID));
            int64_t gmOutBaseOffset = baseOffset * tdPtr_->hOffset + CalcBlockOutOffset();
            DataCopyPad(xOutGM_[gmOutBaseOffset], ubXOut[ubOffset], copyParams);
            ubOffset += vPartCnt * wAlign;
            ubPartBase.SetValue(modPartID, baseOffset + vPartCnt);
        }
    }
    xOutQue_.FreeTensor(ubXOut);
    xInQue_.FreeTensor(ubXIn);
    pR2InQue_.FreeTensor(ubPartIn);
}

template <typename T>
__aicore__ inline void DynPartBase<T>::SinglePartCopyOutX(LocalTensor<uint64_t>& ubPartBase, uint32_t partLen,
                                                          int32_t begPart, int32_t endPart, int64_t gmInBaseOffset)
{
    InsertSync(HardEvent::MTE2_S);
    auto ubXIn = xInQue_.AllocTensor<T>();
    auto ubPartIn = pR2InQue_.DeQue<int32_t>();
    auto wLen = (blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->wMSize : tdPtr_->wTSize;
    int64_t wLpCnt = wLen / tdPtr_->wLpUnit;
    int64_t wLeft = wLen % tdPtr_->wLpUnit;
    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};

    for (uint32_t lpIdx = 0; lpIdx < partLen; ++lpIdx) {
        int32_t partID = ubPartIn.GetValue(lpIdx);
        if (partID >= begPart && partID < endPart) {
            int32_t modPartID = partID % NUM_PARTITION_UNIT;
            int64_t baseOffset = ubPartBase.GetValue(modPartID);
            xOutGM_.SetGlobalBuffer(outGMList_.GetDataPtr<T>(partID));
            int64_t gmOutBaseOffset = baseOffset * tdPtr_->hOffset + CalcBlockOutOffset();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(tdPtr_->wLpUnit * sizeof(T)), 0, 0, 0};
            for (int64_t wLpIdx = 0; wLpIdx < wLpCnt; ++wLpIdx) {
                DataCopyPad(ubXIn, xInGM_[wLpIdx * tdPtr_->wLpUnit + lpIdx * tdPtr_->hOffset + gmInBaseOffset],
                            copyParams, copyPadParams);
                InsertSync(HardEvent::MTE2_MTE3);
                DataCopyPad(xOutGM_[wLpIdx * tdPtr_->wLpUnit + gmOutBaseOffset], ubXIn, copyParams);
                InsertSync(HardEvent::MTE3_MTE2);
            }
            if (wLeft > 0) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(wLeft * sizeof(T)), 0, 0, 0};
                DataCopyPad(ubXIn, xInGM_[wLpCnt * tdPtr_->wLpUnit + lpIdx * tdPtr_->hOffset + gmInBaseOffset],
                            copyParams, copyPadParams);
                InsertSync(HardEvent::MTE2_MTE3);
                DataCopyPad(xOutGM_[wLpCnt * tdPtr_->wLpUnit + gmOutBaseOffset], ubXIn, copyParams);
            }
            InsertSync(HardEvent::MTE3_S);
            ubPartBase.SetValue(modPartID, baseOffset + 1);
        }
    }
    xInQue_.FreeTensor(ubXIn);
    pR2InQue_.FreeTensor(ubPartIn);
}

template <typename T>
__aicore__ inline void DynPartBase<T>::CopyPIn(int64_t hLpIdx, uint32_t hLen)
{
    auto ubR2PIn = pR2InQue_.AllocTensor<int32_t>();
    DataCopyPadExtParams<int32_t> copyPadParams1{false, 0, 0, 0};
    DataCopyExtParams copyParams1{1, static_cast<uint32_t>(hLen * sizeof(int32_t)), 0, 0, 0};
    if (tdPtr_->tilingKey == TILING_W_MC_UB_CAN_HOLD_SPLIT_W ||
        tdPtr_->tilingKey == TILING_W_MC_UB_CANNOT_HOLD_SPLIT_W) {
        DataCopyPad(ubR2PIn, pInGM_[hLpIdx * tdPtr_->hLpUnit], copyParams1, copyPadParams1);
    } else {
        DataCopyPad(ubR2PIn, pInGM_[this->blockIdx_ * tdPtr_->hMSize + hLpIdx * tdPtr_->hLpUnit], copyParams1,
                    copyPadParams1);
    }
    pR2InQue_.EnQue(ubR2PIn);
}

template <typename T>
__aicore__ inline void DynPartBase<T>::CopyXPIn(int64_t hLpIdx, uint32_t hLen)
{
    auto wLen = (this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->wMSize : tdPtr_->wTSize;
    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};
    DataCopyExtParams copyParams{static_cast<uint16_t>(hLen), static_cast<uint32_t>(wLen * sizeof(T)),
                                 static_cast<int64_t>((tdPtr_->hOffset - wLen) * sizeof(T)), 0, 0};
    auto ubXIn = xInQue_.AllocTensor<T>();
    DataCopyPad(ubXIn, xInGM_[gmXInBlockOffset_ + hLpIdx * tdPtr_->hLpUnit * tdPtr_->hOffset], copyParams,
                copyPadParams);
    xInQue_.EnQue(ubXIn);

    CopyPIn(hLpIdx, hLen);
}

template <typename T>
__aicore__ inline void DynPartBase<T>::SinglePartProcess(TQue<QuePosition::VECCALC, 1>& pBaseQue, int32_t begPart,
                                                         int32_t endPart)
{
    InitProcessBuffer();

    int64_t hSize = (this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->hMSize : tdPtr_->hTSize;
    int64_t hLpCnt = hSize / tdPtr_->hLpUnit;
    uint32_t hLeft = static_cast<uint32_t>(hSize % tdPtr_->hLpUnit);
    for (int64_t hLpIdx = 0; hLpIdx < hLpCnt; ++hLpIdx) {
        CopyPIn(hLpIdx, static_cast<uint32_t>(tdPtr_->hLpUnit));
        auto ubPartBase = pBaseQue.DeQue<uint64_t>();
        int64_t gmInBaseOffset = gmXInBlockOffset_ + hLpIdx * tdPtr_->hLpUnit * tdPtr_->hOffset;
        SinglePartCopyOutX(ubPartBase, tdPtr_->hLpUnit, begPart, endPart, gmInBaseOffset);
        pBaseQue.EnQue(ubPartBase);
    }
    if (hLeft > 0U) {
        CopyPIn(hLpCnt, hLeft);
        auto ubPartBase = pBaseQue.DeQue<uint64_t>();
        int64_t gmInBaseOffset = gmXInBlockOffset_ + hLpCnt * tdPtr_->hLpUnit * tdPtr_->hOffset;
        SinglePartCopyOutX(ubPartBase, hLeft, begPart, endPart, gmInBaseOffset);
        pBaseQue.EnQue(ubPartBase);
    }
    bufPoolProcess_.Reset();
}

template <typename T>
__aicore__ inline void DynPartBase<T>::MultiplePartProcess(TQue<QuePosition::VECCALC, 1>& pBaseQue, int32_t begPart,
                                                           int32_t endPart)
{
    InitProcessBuffer();

    int64_t hSize = (this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->hMSize : tdPtr_->hTSize;
    int64_t hLpCnt = hSize / tdPtr_->hLpUnit;
    uint32_t hLeft = static_cast<uint32_t>(hSize % tdPtr_->hLpUnit);
    for (int64_t hLpIdx = 0; hLpIdx < hLpCnt; ++hLpIdx) {
        CopyXPIn(hLpIdx, static_cast<uint32_t>(tdPtr_->hLpUnit));
        auto ubPartBase = pBaseQue.DeQue<uint64_t>();
        MultiplePartCopyOutX(ubPartBase, tdPtr_->hLpUnit, begPart, endPart);
        pBaseQue.EnQue(ubPartBase);
    }
    if (hLeft > 0U) {
        CopyXPIn(hLpCnt, hLeft);
        auto ubPartBase = pBaseQue.DeQue<uint64_t>();
        MultiplePartCopyOutX(ubPartBase, hLeft, begPart, endPart);
        pBaseQue.EnQue(ubPartBase);
    }
    bufPoolProcess_.Reset();
}

template <typename T>
__aicore__ inline void DynPartBase<T>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    switch (event) {
        case HardEvent::V_MTE3:
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            break;
        case HardEvent::V_MTE2:
            SetFlag<HardEvent::V_MTE2>(eventID);
            WaitFlag<HardEvent::V_MTE2>(eventID);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            break;
        case HardEvent::MTE2_V:
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            break;
        case HardEvent::MTE2_MTE3:
            SetFlag<HardEvent::MTE2_MTE3>(eventID);
            WaitFlag<HardEvent::MTE2_MTE3>(eventID);
            break;
        case HardEvent::S_V:
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);
            break;
        case HardEvent::V_S:
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            break;
        case HardEvent::S_MTE3:
            SetFlag<HardEvent::S_MTE3>(eventID);
            WaitFlag<HardEvent::S_MTE3>(eventID);
            break;
        case HardEvent::MTE3_V:
            SetFlag<HardEvent::MTE3_V>(eventID);
            WaitFlag<HardEvent::MTE3_V>(eventID);
            break;
        case HardEvent::MTE3_S:
            SetFlag<HardEvent::MTE3_S>(eventID);
            WaitFlag<HardEvent::MTE3_S>(eventID);
            break;
        case HardEvent::MTE2_S:
            SetFlag<HardEvent::MTE2_S>(eventID);
            WaitFlag<HardEvent::MTE2_S>(eventID);
            break;
        default:
            break;
    }
}

} // namespace DynPart

#endif // OP_KERNEL_DYNAMIC_PARTITION_BASE_H_
