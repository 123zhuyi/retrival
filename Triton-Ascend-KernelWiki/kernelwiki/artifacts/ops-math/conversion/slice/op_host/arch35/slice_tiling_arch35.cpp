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
 * \file slice_tiling_arch35.cpp
 * \brief
 */
#include "slice_tiling_arch35.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {

void SliceTiling::CalMaxSplitDim()
{
    // 受movealignv2的参数限制,blockcout:uint16, srcstride:uint32
    // 输入维度越高，索引生产的scalar越重
    bool isUbinnerGather = lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_ >= MIN_MOVE_ALIGN_LEN &&
                           dimNum_ <= NUMBER_FOUR &&
                           (lastTwoInputDim_ - lastTwoOutputDim_) * lastOneInputDim_ * xDtypeSize_ <= MAX_UINT32_NUM &&
                           lastOneInputDim_ * lastTwoInputDim_ * lastThreeInputDim_ * xDtypeSize_ <= MAX_UINT32_NUM;
    if (dimNum_ == NUMBER_TWO && lastOneOutputDim_ * xDtypeSize_ < VL_SIZE && 
        lastTwoOutputDim_ == lastTwoInputDim_ &&
        VL_SIZE * lastTwoInputDim_ < coreNum_ * ubSize_ / NUMBER_FOUR && 
        (lastOneOutputDim_ != 1 || lastTwoOutputDim_ <= LAST_DIM_MIN_DATA_SIZE)) {
        // 针对2维，仅尾轴切分的小shape场景的性能特化模板
        // 尾轴输入大于等于1个VL长度或者尾轴输入小于1个VL长度，且尾轴为偏移0时，才进入本模板
        maxSplitDim_ = MAX_TWO_DIM_UB_SPLIT_AXIS_NUM;
    } else if (
        (lastOneInputDim_ * xDtypeSize_) % BLOCK_SIZE != 0 && lastOneInputDim_ * xDtypeSize_ <= VL_SIZE &&
        (lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_ >= VL_SIZE ||
         lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_ % BLOCK_SIZE == 0) &&
        lastOneOutputDim_ * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE && totalOutputSize_ > MIN_OUTPUT_SIZE &&
        lastOneInputDim_ / lastOneOutputDim_ < DATA_COPY_SPARSITY_THRESHOLD && isUbinnerGather) {
        // 输入的尾轴非block对齐
        // 输入尾轴<= 256B && 64B <= 输出尾轴 <= 256B
        // 当连续搬入的burstlen小于256B且非block对齐时，性能比compact模式更差
        isUnalignCopy_ = true;
        maxSplitDim_ = MAX_SLICE_GATHER_UB_SPLIT_AXIS_NUM;
        isSliceGather_ = true;
    } else if (lastOneOutputDim_ * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
    } else if (
        lastOneInputDim_ * xDtypeSize_ < RESERVE_LAST_DIM_SIZE &&
        lastOneInputDim_ / lastOneOutputDim_ < DATA_SPARSITY_THRESHOLD &&
        totalOutputSize_ / lastOneOutputDim_ * lastOneInputDim_ >
            ubSize_ * lastOneInputDim_ / (lastOneOutputDim_ + lastOneInputDim_) * coreNum_ &&
        xDtypeSize_ != 1 && isUbinnerGather) {
        // b8 性能不稳定
        // 数据稀疏度越低，搬入的有效数据越少
        // 数据太小，一次搬入就处理完，vf计算完全不能被MTE掩掉的场景
        // 上述2个条件保证了totalOutputSize_ > LAST_DIM_MIN_DATA_SIZE
        maxSplitDim_ = MAX_SLICE_GATHER_UB_SPLIT_AXIS_NUM;
        isSliceGather_ = true;
    } else {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
    }
}

void SliceTiling::SetTilingMode()
{
    if (maxSplitDim_ == MAX_TWO_DIM_UB_SPLIT_AXIS_NUM) {
        SetTwoDimSmallShapeTilingMode();
    } else if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM) {
        SetMovAlignV2TilingMode();
    } else if (isSliceGather_) {
        SetSliceGatherTilingMode();
    } else {
        SetNddmaTilingMode();
    }
}

void SliceTiling::SetTwoDimSmallShapeTilingMode()
{
    tilingKey_ = SLICE_KEY_TWO_DIM_SMALL_SHAPE;
}

/*
input
(A, B, C, D，E)
outputShape:
(a, b, c, d，e)

(a, b, c, d，E)
搬入d*E的数据
场景1：ub切d
(a, b, c, d.o, d.i, E)
burstlen = d.i * E * dype
nburst = 1

场景2：ub切c
(a, b, c.o, c.i, d，E)
burstlen = d * E * dype
nburst = c.i

场景3：ub切b
(a, b.o, b.i, c, d，E)
burstlen = d * E * dype
nburst = c
loop1size = b.i

场景4：ub切a
(a.o, a.i, b, c, d，E)
burstlen = d * E * dype
nburst = c
loop1size = b
loop2size = a.i

维度太高，scalar计算也会很重，可以考虑限制, inputdim <= 4

*/
void SliceTiling::SetSliceGatherTilingMode()
{
    int32_t shapeSize = dimNum_;
    // SliceGather场景的准入条件是尾轴小于64B，保证了尾轴不会被切UB
    if (isUnalignCopy_) {
        tilingKey_ = SLICE_KEY_MOVE_UNALIGN_GATHER;
    } else {
        tilingKey_ = STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER;
    }
    // ubfactor切-2轴
    if (ubIndex_ == shapeSize - NUMBER_TWO) {
        SliceGatherUbSplitLastTwoDim();
        return;
    }

    // ubfactor切-3轴
    if (ubIndex_ == shapeSize - NUMBER_THREE) {
        SliceGatherUbSplitLastThreeDim();
        return;
    }

    // ubfactor切-4轴
    if (ubIndex_ == shapeSize - NUMBER_FOUR) {
        SliceGatherUbSplitLastFourDim();
        return;
    }
}

/*
场景1：ub切d
(a, b, c, d.o, d.i, E)
burstlen = d.i * E * dype
nburst = 1
*/
void SliceTiling::SliceGatherUbSplitLastTwoDim()
{
    mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(1);
    mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(ubFactor_ * lastOneInputDim_ * xDtypeSize_);
    mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(0);
    mainMoveAlignV2Info_.dstStride = static_cast<uint32_t>(0);
    outBlockLen_ = static_cast<uint32_t>(ubFactor_ * lastOneOutputDim_ * xDtypeSize_);
}

/*
场景2：ub切c
(a, b, c.o, c.i, d，E)
burstlen = d * E * dype
nburst = c.i
*/
void SliceTiling::SliceGatherUbSplitLastThreeDim()
{
    mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(ubFactor_);
    mainMoveAlignV2Info_.blockLen =
        static_cast<uint32_t>(lastTwoOutputDim_ * lastOneInputDim_ * xDtypeSize_); // 每个blockLen之间会补pad
    mainMoveAlignV2Info_.srcStride =
        static_cast<uint32_t>((lastTwoInputDim_ - lastTwoOutputDim_) * lastOneInputDim_ * xDtypeSize_);
    mainMoveAlignV2Info_.dstStride = static_cast<uint32_t>(0);

    // out
    outBlockLen_ = static_cast<uint32_t>(lastTwoOutputDim_ * lastOneOutputDim_ * xDtypeSize_);
}

/*
场景3：ub切b
(a, b.o, b.i, c, d，E)
burstlen = d * E * dype
nburst = c
loop1size = b.i
*/
void SliceTiling::SliceGatherUbSplitLastFourDim()
{
    mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastThreeOutputDim_);
    mainMoveAlignV2Info_.blockLen =
        static_cast<uint32_t>(lastTwoOutputDim_ * lastOneInputDim_ * xDtypeSize_); // 每个blockLen之间会补pad
    mainMoveAlignV2Info_.srcStride =
        static_cast<uint32_t>((lastTwoInputDim_ - lastTwoOutputDim_) * lastOneInputDim_ * xDtypeSize_);
    mainMoveAlignV2Info_.dstStride = static_cast<uint32_t>(0);
    outBlockLen_ = static_cast<uint32_t>(lastTwoOutputDim_ * lastOneOutputDim_ * xDtypeSize_);

    int64_t loop1SrcStride = lastOneInputDim_ * lastTwoInputDim_ * lastThreeInputDim_ * xDtypeSize_;
    int64_t loop1DstStride =
        Ops::Base::CeilAlign(lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_, BLOCK_SIZE) * lastThreeOutputDim_;
    mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(ubFactor_);
    mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride); // repeat stride是头和头之间的间隔
    mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
}

void SliceTiling::CalSliceRowsStepsParams()
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - static_cast<int32_t>(NUMBER_TWO); i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    // ub非最后一根轴
    if (ubIndex_ != static_cast<int64_t>(dimNum_) - static_cast<int64_t>(1)) {
        inLoopSteps_ = ubFactor_ * inputSteps_[ubIndex_ + 1];
        outLoopSteps_ = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps_ = ubFactor_ * strides_[ubIndex_];
        outLoopSteps_ = ubFactor_;
    }
}

void SliceTiling::SetRowsStepsParamsFor150(SliceMoveAlignLast2DimTilingData& tilingData)
{
    CalSliceRowsStepsParams();
    tilingData.ubInLoopSteps = inLoopSteps_;
    tilingData.ubOutLoopSteps = outLoopSteps_;
    for (size_t i = 0; i < NUMBER_TWO; i++) {
        tilingData.inputSteps[i] = inputSteps_[i];
    }
}

void SliceTiling::SetShortMoveAlignParams(SliceMoveAlignParams& params, const MoveAlignV2Info& actInfo)
{
    params.blockCount = actInfo.blockCount;
    params.blockLen = actInfo.blockLen;
    params.srcStride = actInfo.srcStride;
    params.dstStride = actInfo.dstStride;
}

void SliceTiling::SetMoveAlignParamsSlice(StridedSliceMoveAlignParams2& params, const MoveAlignV2Info& actInfo)
{
    params.blockCount = actInfo.blockCount;
    params.blockLen = actInfo.blockLen;
    params.srcStride = actInfo.srcStride;
    params.dstStride = actInfo.dstStride;
    params.loop1Size = actInfo.loop1Size;
    params.loop2Size = actInfo.loop2Size;
    params.loop1SrcStride = actInfo.loop1SrcStride;
    params.loop1DstStride = actInfo.loop1DstStride;
    params.loop2SrcStride = actInfo.loop2SrcStride;
    params.loop2DstStride = actInfo.loop2DstStride;
}

void SliceTiling::SetRowsStepsParamsSlice(StridedSliceTilingData2& tilingData)
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - NUMBER_TWO; i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        tilingData.rowsOffsetSteps[i] = rowsOffsetSteps_[i];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        tilingData.inputSteps[i] = inputSteps_[i];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    int64_t inLoopSteps = 1;
    int64_t outLoopSteps = 1;
    // ub非最后一根轴
    if (ubIndex_ != dimNum_ - 1) {
        inLoopSteps = ubFactor_ * strides_[ubIndex_] * inputSteps_[ubIndex_ + 1];
        outLoopSteps = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps = ubFactor_ * strides_[ubIndex_];
        outLoopSteps = ubFactor_;
    }
    tilingData.ubInLoopSteps = inLoopSteps;
    tilingData.ubOutLoopSteps = outLoopSteps;
}

void SliceTiling::FillSliceTilingData150()
{
    size_t tilingDataSize = sizeof(SliceMoveAlignLast2DimTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData150.");
    for (size_t i = 0; i < NUMBER_TWO; i++) {
        sliceMoveAlignLast2DimTilingData_.begin[i] = begin_[i];
        sliceMoveAlignLast2DimTilingData_.outputShape[i] = outputShape_[i];
    }
    sliceMoveAlignLast2DimTilingData_.ubSize = ubSize_;
    sliceMoveAlignLast2DimTilingData_.blkFactor = blkFactor_;
    sliceMoveAlignLast2DimTilingData_.blkTailFactor = blkTailFactor_;
    sliceMoveAlignLast2DimTilingData_.ubFactor = ubFactor_;
    sliceMoveAlignLast2DimTilingData_.ubTailFactor = ubTailFactor_;
    sliceMoveAlignLast2DimTilingData_.ubTailTailFactor = ubTailTailFactor_;
    sliceMoveAlignLast2DimTilingData_.realCoreNum = realCoreNum_;
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    sliceMoveAlignLast2DimTilingData_.isBeginConst = isConstBegin;

    SetShortMoveAlignParams(sliceMoveAlignLast2DimTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    SetRowsStepsParamsFor150(sliceMoveAlignLast2DimTilingData_);

    auto tilingData = tilingContext_->GetTilingData<SliceMoveAlignLast2DimTilingData>();
    errno_t ret = memcpy_s(
        tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceMoveAlignLast2DimTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceBaseTilingData(SliceBaseTilingData& tilingData)
{
    tilingData.ubSize = ubSize_;
    tilingData.ubIndex = ubIndex_;
    tilingData.ubFactor = ubFactor_;
    tilingData.ubTailFactor = ubTailFactor_;
    tilingData.ubTailTailFactor = ubTailTailFactor_;
    tilingData.realCoreNum = realCoreNum_;
    tilingData.inputDims = dimNum_;
    tilingData.blkIndex = blkIndex_;
    tilingData.blkFactor = blkFactor_;
    tilingData.blkTailFactor = blkTailFactor_;
    for (int i = 0; i < MAX_AXIS_NUM_FOR_STRIDESLICE; ++i) {
        tilingData.begin[i] = begin_[i];
        tilingData.outputShape[i] = outputShape_[i];
    }
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    tilingData.isBeginConst = isConstBegin;
    CalSliceRowsStepsParams();
    tilingData.ubInLoopSteps = inLoopSteps_;
    for (int i = 0; i < MAX_AXIS_NUM_FOR_STRIDESLICE; ++i) {
        tilingData.inputSteps[i] = inputSteps_[i];
        tilingData.rowsOffsetSteps[i] = rowsOffsetSteps_[i];
    }
}

void SliceTiling::FillSliceTilingData100()
{
    size_t tilingDataSize = sizeof(SliceMoveAlignTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData100.");
    FillSliceBaseTilingData(sliceMoveAlignTilingData_.sliceBaseTilingData);

    sliceMoveAlignTilingData_.ubOutLoopSteps = outLoopSteps_;
    SetMoveAlignParamsSlice(sliceMoveAlignTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    auto tilingData = tilingContext_->GetTilingData<SliceMoveAlignTilingData>();
    errno_t ret =
        memcpy_s(tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceMoveAlignTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceTilingData101()
{
    size_t tilingDataSize = sizeof(SliceMoveAlignLastDimTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData101.");
    FillSliceBaseTilingData(sliceMoveAlignLastDimTilingData_.sliceBaseTilingData);
    auto tilingData = tilingContext_->GetTilingData<SliceMoveAlignLastDimTilingData>();
    errno_t ret = memcpy_s(
        tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceMoveAlignLastDimTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceTilingData102()
{
    size_t tilingDataSize = sizeof(SliceNDDMATilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData102.");
    FillSliceBaseTilingData(sliceNDDMATilingData_.sliceBaseTilingData);

    sliceNDDMATilingData_.ubOutLoopSteps = outLoopSteps_;
    sliceNDDMATilingData_.nddmaTotalNum = nddmaTotalNum_;
    for (int i = 0; i < MAX_NDDMA_UB_SPLIT_AXIS_NUM; ++i) {
        sliceNDDMATilingData_.nddmaLoopSize[i] = nddmaLoopSize_[i];
        sliceNDDMATilingData_.nddmaLoopSrcStride[i] = nddmaLoopSrcStride_[i];
        sliceNDDMATilingData_.nddmaLoopDstStride[i] = nddmaLoopDstStride_[i];
    }
    auto tilingData = tilingContext_->GetTilingData<SliceNDDMATilingData>();
    errno_t ret = memcpy_s(tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceNDDMATilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceTilingData103()
{
    size_t tilingDataSize = sizeof(SliceNDDMALastDimTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData103.");
    FillSliceBaseTilingData(sliceNDDMALastDimTilingData_.sliceBaseTilingData);

    for (int i = 0; i < MAX_NDDMA_UB_SPLIT_AXIS_NUM; ++i) {
        sliceNDDMALastDimTilingData_.nddmaLoopSrcStride[i] = nddmaLoopSrcStride_[i];
        sliceNDDMALastDimTilingData_.nddmaLoopDstStride[i] = nddmaLoopDstStride_[i];
    }
    auto tilingData = tilingContext_->GetTilingData<SliceNDDMALastDimTilingData>();
    errno_t ret =
        memcpy_s(tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceNDDMALastDimTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceTilingData300()
{
    size_t tilingDataSize = sizeof(SliceMoveAlignGatherTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData300.");
    FillSliceBaseTilingData(sliceMoveAlignGatherTilingData_.sliceBaseTilingData);
    sliceMoveAlignGatherTilingData_.sliceBaseTilingData.ubSize = ubSizeOutput_;
    sliceMoveAlignGatherTilingData_.ubSizeInput = ubSizeInput_;
    sliceMoveAlignGatherTilingData_.lastOneInputDim = lastOneInputDim_;
    sliceMoveAlignGatherTilingData_.outBlockLen = outBlockLen_;
    sliceMoveAlignGatherTilingData_.ubOutLoopSteps = outLoopSteps_;

    SetMoveAlignParamsSlice(sliceMoveAlignGatherTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    auto tilingData = tilingContext_->GetTilingData<SliceMoveAlignGatherTilingData>();
    errno_t ret =
        memcpy_s(tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceMoveAlignGatherTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceTilingData400()
{
    size_t tilingDataSize = sizeof(SliceTwoDimSmallSapeTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData400.");
    sliceTwoDimSmallSapeTilingData_.ubSize = (uint32_t)ubSize_;
    sliceTwoDimSmallSapeTilingData_.realCoreNum = realCoreNum_;
    sliceTwoDimSmallSapeTilingData_.mainCoreNum = mainCoreNum_;
    sliceTwoDimSmallSapeTilingData_.blockLen = outBlockLen_;
    sliceTwoDimSmallSapeTilingData_.blkFactor = blkFactor_;
    sliceTwoDimSmallSapeTilingData_.lastOneInputDim = lastOneInputDim_;
    sliceTwoDimSmallSapeTilingData_.lastOneOutputDim = lastOneOutputDim_;
    sliceTwoDimSmallSapeTilingData_.lastOneDimOffset = begin_[1];
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    sliceTwoDimSmallSapeTilingData_.isBeginConst = isConstBegin;

    auto tilingData = tilingContext_->GetTilingData<SliceTwoDimSmallSapeTilingData>();
    errno_t ret =
        memcpy_s(tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceTwoDimSmallSapeTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillSliceTilingDataOther()
{
    size_t tilingDataSize = sizeof(SliceTilingData);
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData.");
    sliceTilingData_.stridedSliceTilingData.ubSize = ubSize_;
    sliceTilingData_.stridedSliceTilingData.coreNum = coreNum_;
    sliceTilingData_.stridedSliceTilingData.ubIndex = ubIndex_;
    sliceTilingData_.stridedSliceTilingData.ubFactor = ubFactor_;
    sliceTilingData_.stridedSliceTilingData.ubTailFactor = ubTailFactor_;
    sliceTilingData_.stridedSliceTilingData.ubTailTailFactor = ubTailTailFactor_;
    sliceTilingData_.stridedSliceTilingData.realCoreNum = realCoreNum_;
    sliceTilingData_.stridedSliceTilingData.inputDims = dimNum_;
    sliceTilingData_.stridedSliceTilingData.blkIndex = blkIndex_;
    sliceTilingData_.stridedSliceTilingData.blkFactor = blkFactor_;
    sliceTilingData_.stridedSliceTilingData.blkTailFactor = blkTailFactor_;
    sliceTilingData_.stridedSliceTilingData.xDtypeSize = xDtypeSize_;
    sliceTilingData_.stridedSliceTilingData.tilingKey = tilingKey_;
    sliceTilingData_.stridedSliceTilingData.nddmaTotalNum = nddmaTotalNum_;
    for (int i = 0; i < MAX_NDDMA_UB_SPLIT_AXIS_NUM; ++i) {
        sliceTilingData_.stridedSliceTilingData.nddmaLoopSize[i] = nddmaLoopSize_[i];
        sliceTilingData_.stridedSliceTilingData.nddmaLoopSrcStride[i] = nddmaLoopSrcStride_[i];
        sliceTilingData_.stridedSliceTilingData.nddmaLoopDstStride[i] = nddmaLoopDstStride_[i];
        sliceTilingData_.stridedSliceTilingData.outputShapeProd[i] = outputShapeProd_[i];
        sliceTilingData_.stridedSliceTilingData.inputShapeProd[i] = inputShapeProd_[i];
    }
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    sliceTilingData_.stridedSliceTilingData.isBeginConst = isConstBegin;

    SetMoveAlignParamsSlice(sliceTilingData_.stridedSliceTilingData.moveAlignParams, mainMoveAlignV2Info_);

    for (size_t i = 0; i < sliceParam_.inputShape.GetDimNum(); i++) {
        sliceTilingData_.stridedSliceTilingData.begin[i] = sliceParam_.beginList.GetDim(i);
        end_[i] = sliceParam_.endList.GetDim(i);
        sliceTilingData_.stridedSliceTilingData.strides[i] = sliceParam_.strideList.GetDim(i);
        inputShape_[i] = sliceParam_.inputShape.GetDim(i);
        sliceTilingData_.stridedSliceTilingData.outputShape[i] = sliceParam_.outputShape.GetDim(i);
    }

    SetRowsStepsParamsSlice(sliceTilingData_.stridedSliceTilingData);
    auto tilingData = tilingContext_->GetTilingData<SliceTilingData>();
    errno_t ret = memcpy_s(tilingData, tilingDataSize, reinterpret_cast<void*>(&sliceTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    }
}

void SliceTiling::FillTilingData()
{
    for (size_t i = 0; i < sliceParam_.inputShape.GetDimNum(); i++) {
        begin_[i] = sliceParam_.beginList.GetDim(i);
        end_[i] = sliceParam_.endList.GetDim(i);
        inputShape_[i] = sliceParam_.inputShape.GetDim(i);
        outputShape_[i] = sliceParam_.outputShape.GetDim(i);
    }
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            FillSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            FillSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
            FillSliceTilingData102();
            break;
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
            FillSliceTilingData103();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            FillSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
        case SLICE_KEY_MOVE_UNALIGN_GATHER:
            FillSliceTilingData300();
            break;
        case SLICE_KEY_TWO_DIM_SMALL_SHAPE:
            FillSliceTilingData400();
            break;
        default:
            FillSliceTilingDataOther();
            break;
    }
}

void SliceTiling::PrintSliceTilingDataOther()
{
    StridedSliceTilingData2& tilingData = sliceTilingData_.stridedSliceTilingData;
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is ubSize:%ld, coreNum:%ld, realCoreNum:%ld, \
            ubIndex:%ld, ubFactor:%ld, ubTailFactor:%ld, ubTailTailFactor:%ld, \
            blkIndex:%ld, blkFactor:%ld, blkTailFactor:%ld, xDtypeSize:%ld, tilingKey:%ld, \
            isBeginConst:%d, begin:%s, end:%s, stride:%s, inputShape:%s, outputShape:%s, \
            rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s",
        tilingData.ubSize, tilingData.coreNum, tilingData.realCoreNum, tilingData.ubIndex, tilingData.ubFactor,
        tilingData.ubTailFactor, tilingData.ubTailTailFactor, tilingData.blkIndex, tilingData.blkFactor,
        tilingData.blkTailFactor, tilingData.xDtypeSize, tilingData.tilingKey, tilingData.isBeginConst,
        ArrayToStr(tilingData.begin, dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(tilingData.strides, dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(tilingData.outputShape, dimNum_).c_str(), ArrayToStr(tilingData.rowsOffsetSteps, dimNum_).c_str(),
        ArrayToStr(tilingData.inputSteps, dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str());

    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is nddmaTotalNum:%ld nddmaLoopSize:%s, nddmaLoopSrcStride: %s, \
            nddmaLoopDstStride: %s, moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
            loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u outputShapeProd: %s \
            inputShapeProd: %s Tiling4StrideSlice ends.",
        tilingData.nddmaTotalNum, ArrayToStr(tilingData.nddmaLoopSize, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData.nddmaLoopSrcStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData.nddmaLoopDstStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(), mainMoveAlignV2Info_.blockCount,
        mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride,
        mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride,
        mainMoveAlignV2Info_.loop1DstStride, mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride,
        ArrayToStr(tilingData.outputShapeProd, MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData.inputShapeProd, MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str());
}

void SliceTiling::PrintTilingData()
{
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            PrintSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            PrintSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
            PrintSliceTilingData102();
            break;
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
            PrintSliceTilingData103();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            PrintSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
            PrintSliceTilingData300();
            break;
        case SLICE_KEY_TWO_DIM_SMALL_SHAPE:
            PrintSliceTilingData400();
            break;
        default:
            PrintSliceTilingDataOther();
            break;
    }
}

void SliceTiling::PrintSliceBaseTilingData(SliceBaseTilingData& tilingData)
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceBaseTilingData is ubSize:%d, realCoreNum:%d, \
            ubIndex:%d, ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkIndex:%d, blkFactor:%ld, blkTailFactor:%ld, \
            isBeginConst:%d, begin:%s, end:%s, inputShape:%s, outputShape:%s, \
            rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld",
        tilingData.ubSize, tilingData.realCoreNum, tilingData.ubIndex, tilingData.ubFactor, tilingData.ubTailFactor,
        tilingData.ubTailTailFactor, tilingData.blkIndex, tilingData.blkFactor, tilingData.blkTailFactor,
        tilingData.isBeginConst, ArrayToStr(tilingData.begin, dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(inputShape_, dimNum_).c_str(), ArrayToStr(tilingData.outputShape, dimNum_).c_str(),
        ArrayToStr(tilingData.rowsOffsetSteps, dimNum_).c_str(), ArrayToStr(tilingData.inputSteps, dimNum_).c_str(),
        ArrayToStr(outputSteps_, dimNum_).c_str(), tilingData.ubInLoopSteps);
}

void SliceTiling::PrintSliceTilingData150()
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceMoveAlignLast2DimTilingData is ubSize:%d, realCoreNum:%d, \
            ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkFactor:%ld, blkTailFactor:%ld, \
            isBeginConst:%d, begin:%s, end:%s, inputShape:%s, outputShape:%s, \
            inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld, ubOutLoopSteps:%ld, \
            moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u",
        sliceMoveAlignLast2DimTilingData_.ubSize, sliceMoveAlignLast2DimTilingData_.realCoreNum,
        sliceMoveAlignLast2DimTilingData_.ubFactor, sliceMoveAlignLast2DimTilingData_.ubTailFactor,
        sliceMoveAlignLast2DimTilingData_.ubTailTailFactor, sliceMoveAlignLast2DimTilingData_.blkFactor,
        sliceMoveAlignLast2DimTilingData_.blkTailFactor, sliceMoveAlignLast2DimTilingData_.isBeginConst,
        ArrayToStr(sliceMoveAlignLast2DimTilingData_.begin, dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(sliceMoveAlignLast2DimTilingData_.outputShape, dimNum_).c_str(),
        ArrayToStr(sliceMoveAlignLast2DimTilingData_.inputSteps, dimNum_).c_str(),
        ArrayToStr(outputSteps_, dimNum_).c_str(), sliceMoveAlignLast2DimTilingData_.ubInLoopSteps,
        sliceMoveAlignLast2DimTilingData_.ubOutLoopSteps, mainMoveAlignV2Info_.blockCount,
        mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride);
}

void SliceTiling::PrintSliceTilingData101()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceMoveAlignLastDimTilingData:");
    PrintSliceBaseTilingData(sliceMoveAlignLastDimTilingData_.sliceBaseTilingData);
}

void SliceTiling::PrintSliceTilingData100()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceMoveAlignTilingData:");
    PrintSliceBaseTilingData(sliceMoveAlignTilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceMoveAlignTilingData is ubOutLoopSteps:%ld, \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        sliceMoveAlignTilingData_.ubOutLoopSteps, mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen,
        mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size,
        mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
        mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void SliceTiling::PrintSliceTilingData102()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceNDDMATilingData:");
    PrintSliceBaseTilingData(sliceNDDMATilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceNDDMATilingData is ubOutLoopSteps:%ld, \
    nddmaTotalNum:%ld, nddmaLoopSize:%s, nddmaLoopSrcStride: %s, nddmaLoopDstStride: %s",
        sliceNDDMATilingData_.ubOutLoopSteps, sliceNDDMATilingData_.nddmaTotalNum,
        ArrayToStr(sliceNDDMATilingData_.nddmaLoopSize, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(sliceNDDMATilingData_.nddmaLoopSrcStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(sliceNDDMATilingData_.nddmaLoopDstStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str());
}

void SliceTiling::PrintSliceTilingData103()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceNDDMALastDimTilingData:");
    PrintSliceBaseTilingData(sliceNDDMALastDimTilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(), "SliceNDDMALastDimTilingData is nddmaLoopSrcStride: %s, nddmaLoopDstStride: %s",
        ArrayToStr(sliceNDDMALastDimTilingData_.nddmaLoopSrcStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(sliceNDDMALastDimTilingData_.nddmaLoopDstStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str());
}

void SliceTiling::PrintSliceTilingData300()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceMoveAlignGatherTilingData:");
    PrintSliceBaseTilingData(sliceMoveAlignGatherTilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceMoveAlignGatherTilingData is ubOutLoopSteps:%ld, \
    ubSizeInput:%d, lastOneInputDim:%u, outBlockLen:%u, \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        sliceMoveAlignGatherTilingData_.ubOutLoopSteps, sliceMoveAlignGatherTilingData_.ubSizeInput,
        sliceMoveAlignGatherTilingData_.lastOneInputDim, sliceMoveAlignGatherTilingData_.outBlockLen,
        mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
        mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
        mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride, mainMoveAlignV2Info_.loop2SrcStride,
        mainMoveAlignV2Info_.loop2DstStride);
}

void SliceTiling::PrintSliceTilingData400()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceTwoDimSmallSapeTilingData:");
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceTwoDimSmallSapeTilingData is realCoreNum:%d, \
    mainCoreNum:%d, outBlockLen:%d, blkFactor:%d, lastOneInputDim:%ld, lastOneOutputDim:%ld, ubSize:%d, \
    lastOneDimOffset:%ld, isBeginConst:%d",
        sliceTwoDimSmallSapeTilingData_.realCoreNum, sliceTwoDimSmallSapeTilingData_.mainCoreNum,
        sliceTwoDimSmallSapeTilingData_.blockLen, sliceTwoDimSmallSapeTilingData_.blkFactor,
        sliceTwoDimSmallSapeTilingData_.lastOneInputDim, sliceTwoDimSmallSapeTilingData_.lastOneOutputDim,
        sliceTwoDimSmallSapeTilingData_.ubSize, sliceTwoDimSmallSapeTilingData_.lastOneDimOffset,
        sliceTwoDimSmallSapeTilingData_.isBeginConst);
}

void SliceTiling::SetBlockDimAndTilingKey()
{
    tilingContext_->SetBlockDim(realCoreNum_);
    tilingContext_->SetTilingKey(tilingKey_);
}

ge::graphStatus SliceTilingForAscendC(
    gert::TilingContext* context, int64_t coreNum, int64_t ubSize, int64_t cacheLineSize, SliceParasRuntime2& param,
    const ge::DataType dtype)
{
    SliceTiling tilingImpl(context);
    SliceParametersRuntime2 sliceParam;
    sliceParam.inputShape = param.input;
    sliceParam.outputShape = param.output_shape;
    sliceParam.beginList = param.begin_list;
    sliceParam.endList = param.end_list;
    sliceParam.strideList = param.stride_list;
    sliceParam.tilingMode = param.tiling_mode;
    sliceParam.coreNum = param.core_num;
    sliceParam.isBeginConst = param.is_begin_const;
    OP_LOGI(context->GetNodeName(), "slice params: %s", sliceParam.to_string().c_str());

    if (tilingImpl.Init(coreNum, ubSize, cacheLineSize, sliceParam, dtype) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "SliceTilingForAscendC init failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingImpl.RunStrideSliceTiling();
}

template <typename T>
static ge::graphStatus AssignInputValueOpt(
    gert::TilingContext* context, size_t size, const gert::Tensor* tensor, gert::Shape& list_vector, bool isAscendc,
    bool& isConst, const ge::DataType dtype)
{
    list_vector.SetDimNum(size);
    const T* value = tensor->GetData<T>();
    if (value == nullptr && isAscendc) {
        OP_LOGI(context->GetNodeName(), "ascendc get const data nullptr");
        isConst = false;
        for (size_t i = 0; i < size; i++) {
            list_vector[i] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (value == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "value", "nullptr",
                "When check input is const or not, const value cannot be nullptr");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < size; i++) {
        list_vector[i] = value[i];
    }
    if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        if (list_vector[size - 1] & 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "offsets", Ops::Base::ToString(list_vector).c_str(),
                    "Expected last dimension of offsets to be even for fp4 input.");
            return ge::GRAPH_FAILED;
        }
        list_vector[size - 1] /= SLICE_CONST2;
    }
    isConst = true;
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus AssignInputValue(
    gert::TilingContext* context, size_t size, const gert::Tensor* tensor, gert::Shape& list_vector,
    const ge::DataType dtype)
{
    int32_t dim_num = tensor->GetShapeSize();
    list_vector.SetDimNum(dim_num);
    const T* value = tensor->GetData<T>();
    if (value == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "value", "nullptr",
                "When check input is const or not, const value cannot be nullptr");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < size; i++) {
        list_vector[i] = value[i];
    }
    if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        if (list_vector[size - 1] & 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "last dimension of size", Ops::Base::ToString(list_vector).c_str(),
                    "Expected last dimension of size to be even for fp4 input");
            return ge::GRAPH_FAILED;
        }
        list_vector[size - 1] /= SLICE_CONST2;
    }
    return ge::GRAPH_SUCCESS;
}

static bool CalcEndAndBeginList(
    gert::Shape& list_end_vector, gert::Shape& list_begin_vector, const gert::Shape& shape_input, size_t size,
    bool flag, bool is_begin_const)
{
    if (flag) {
        for (size_t index = 0; index < size; index++) {
            if (list_end_vector[index] == -1) {
                if (is_begin_const == false) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            "Slice", "end", "-1",
                            "End cannot be -1 while begin is not const.");
                    return false;
                }
                list_end_vector[index] = shape_input.GetDim(index) - list_begin_vector[index];
            }
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            if (list_begin_vector[i] < 0 || list_begin_vector[i] + list_end_vector[i] < list_begin_vector[i] ||
                list_begin_vector[i] + list_end_vector[i] > shape_input.GetDim(i)) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            "Slice", "list_begin_vector, list_end_vector, shape_input", 
                            "index: " + std::to_string(i) + ", begin: " + std::to_string(list_begin_vector[i]) +
                            ", end: " + std::to_string(list_end_vector[i])  +
                            ", input shape: " + std::to_string(shape_input.GetDim(i)),
                            "Requirements: 0<=offsets[i]<= offsets[i]+size[i]<=input_shape[i].");
                return false;
            }
        }
    }
    return true;
}

static void MakePerformanceParams(SliceParasRuntime2& parameters)
{
    if (!parameters.is_begin_const) {
        return;
    }
    SliceParasRuntime2 perf_params;
    perf_params.is_begin_const = parameters.is_begin_const;
    bool last_same = false;
    size_t perf_size = 0;
    for (size_t i = 0; i < parameters.input.GetDimNum(); i++) {
        const auto input_shape_i = parameters.input[i];
        const auto output_shape_i = parameters.output_shape[i];
        const auto begin_i = parameters.begin_list[i];
        const auto end_i = parameters.end_list[i];
        const auto stride_i = parameters.stride_list[i];
        if (input_shape_i != output_shape_i || !last_same) {
            perf_params.input.AppendDim(input_shape_i);
            perf_params.output_shape.AppendDim(output_shape_i);
            perf_params.begin_list.AppendDim(begin_i);
            perf_params.end_list.AppendDim(end_i);
            perf_params.stride_list.AppendDim(stride_i);
            last_same = (input_shape_i == output_shape_i);
            perf_size++;
            continue;
        }

        const auto perf_index = perf_size - static_cast<size_t>(1);
        perf_params.input[perf_index] *= input_shape_i;
        perf_params.output_shape[perf_index] *= output_shape_i;
        perf_params.begin_list[perf_index] = 0;
        perf_params.end_list[perf_index] = perf_params.input[perf_index];
        perf_params.stride_list[perf_index] = 1;
    }
    const size_t last_second = 2;
    if (perf_params.input.GetDimNum() > 1 &&
        perf_params.input.GetDim(perf_params.input.GetDimNum() - 1) ==
            perf_params.output_shape.GetDim(perf_params.output_shape.GetDimNum() - 1) &&
        perf_params.stride_list[perf_params.input.GetDimNum() - last_second] == 1) {
        const auto last_second_index = perf_params.input.GetDimNum() - last_second;
        perf_params.input[last_second_index] *= perf_params.input.GetDim(perf_params.input.GetDimNum() - 1);
        perf_params.output_shape[last_second_index] *=
            perf_params.output_shape.GetDim(perf_params.output_shape.GetDimNum() - 1);
        perf_params.begin_list[last_second_index] *= perf_params.input.GetDim(perf_params.input.GetDimNum() - 1);
        perf_params.end_list[last_second_index] *= perf_params.input.GetDim(perf_params.input.GetDimNum() - 1);
        perf_params.stride_list[last_second_index] = 1;

        perf_params.input.SetDimNum(perf_params.input.GetDimNum() - 1);
        perf_params.output_shape.SetDimNum(perf_params.output_shape.GetDimNum() - 1);
        perf_params.begin_list.SetDimNum(perf_params.begin_list.GetDimNum() - 1);
        perf_params.end_list.SetDimNum(perf_params.end_list.GetDimNum() - 1);
        perf_params.stride_list.SetDimNum(perf_params.stride_list.GetDimNum() - 1);
    }

    parameters = perf_params;
}

static ge::graphStatus Tiling4Slice(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const SliceCompileParam*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    const gert::Shape& in_shape = Ops::Base::EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    if (compile_info->block_dim == 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "core number", "0",
                "The vale of core number cannot be 0.");
        return ge::GRAPH_FAILED;
    }
    // instantiate param
    SliceParasRuntime2 sliceparam;
    sliceparam.input = in_shape;
    // get input desc
    auto shape_tensor_x = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor_x);
    // get offsets const value
    auto shape_tensor_offsets = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor_offsets);
    auto shape_size_offsets = static_cast<size_t>(shape_tensor_offsets->GetShapeSize());
    // get size const value
    auto shape_tensor_size = context->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor_size);
    auto shape_size_size = static_cast<size_t>(shape_tensor_size->GetShapeSize());
    // size must be equal
    if (shape_size_offsets != shape_size_size) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "size of offsets, size of shape",
            std::to_string(shape_size_offsets) + ", " + std::to_string(shape_size_size),
            "Length of input_shape, offsets and size must be equal.");
        return ge::GRAPH_FAILED;
    }
    if (shape_size_offsets != in_shape.GetDimNum()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "size of offsets, inout shape size",
            std::to_string(shape_size_offsets) + ", " + std::to_string(in_shape.GetDimNum()),
            "Length of input_shape, offsets and size must be equal.");
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputDtype = shape_tensor_x->GetDataType();
    if (inputDtype == ge::DT_FLOAT4_E2M1 || inputDtype == ge::DT_FLOAT4_E1M2) {
        if (sliceparam.input[shape_size_offsets - 1] & 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "last dimension of input", Ops::Base::ToString(sliceparam.input).c_str(),
                "Expected last dimension of input to be even for fp4 input.");
            return ge::GRAPH_FAILED;
        }
        sliceparam.input[shape_size_offsets - 1] /= SLICE_CONST2;
    }
    ge::DataType offset_dtype = shape_tensor_offsets->GetDataType();
    if (offset_dtype == ge::DT_INT32) {
        // Get offset const val
        OP_CHECK_IF(
            AssignInputValueOpt<int32_t>(
                context, shape_size_offsets, shape_tensor_offsets, sliceparam.begin_list, compile_info->isAscendc,
                sliceparam.is_begin_const, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get offset fail, check input is const or not."), return ge::GRAPH_FAILED);
    } else {
        // Get offset const val
        OP_CHECK_IF(
            AssignInputValueOpt<int64_t>(
                context, shape_size_offsets, shape_tensor_offsets, sliceparam.begin_list, compile_info->isAscendc,
                sliceparam.is_begin_const, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get offset fail, check input is const or not."), return ge::GRAPH_FAILED);
    }
    ge::DataType sizeDtype = shape_tensor_size->GetDataType();
    if (sizeDtype == ge::DT_INT32) {
        // Get size const val
        OP_CHECK_IF(
            AssignInputValue<int32_t>(context, shape_size_size, shape_tensor_size, sliceparam.end_list, inputDtype) !=
                ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get size fail, check input is const or not."), return ge::GRAPH_FAILED);
    } else {
        // Get size const val
        OP_CHECK_IF(
            AssignInputValue<int64_t>(context, shape_size_size, shape_tensor_size, sliceparam.end_list, inputDtype) !=
                ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get size fail, check input is const or not."), return ge::GRAPH_FAILED);
    }
    // calc endlist
    bool end_list_flag = true;
    bool isEndValid = CalcEndAndBeginList(
        sliceparam.end_list, sliceparam.begin_list, sliceparam.input, shape_size_offsets, end_list_flag,
        sliceparam.is_begin_const);
    bool begin_list_flag = false;
    bool isBeginValid = CalcEndAndBeginList(
        sliceparam.end_list, sliceparam.begin_list, sliceparam.input, shape_size_size, begin_list_flag,
        sliceparam.is_begin_const);
    OP_CHECK_IF(
        (compile_info->isAscendc && (!isEndValid || !isBeginValid)), OP_LOGE("Slice", "CalcEndAndBeginList failed"),
        return ge::GRAPH_FAILED);

    // in slice end_list is size values
    sliceparam.output_shape = sliceparam.end_list;
    // calc end list
    for (size_t i = 0; i < shape_size_offsets; i++) {
        sliceparam.end_list[i] += sliceparam.begin_list[i];
    }
    sliceparam.stride_list.SetDimNum(shape_size_size);
    for (size_t i = 0; i < shape_size_size; i++) {
        sliceparam.stride_list[i] = 1;
    }
    MakePerformanceParams(sliceparam);
    if (inputDtype == ge::DT_FLOAT4_E2M1 || inputDtype == ge::DT_FLOAT4_E1M2) {
        inputDtype = ge::DT_INT8;
    }
    return SliceTilingForAscendC(
        context, compile_info->block_dim, compile_info->ub_size, compile_info->cacheLineSize, sliceparam, inputDtype);
}

static ge::graphStatus TilingPrepare4Slice(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4Slice.");
    auto compileInfo = context->GetCompiledInfo<SliceCompileParam>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->block_dim = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->block_dim <= 0), OP_LOGE(context->GetNodeName(), "block_dim invalid."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ub_size <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."), return ge::GRAPH_FAILED);

    compileInfo->isAscendc = true;
    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->cacheLineSize == static_cast<uint32_t>(0)),
        OP_LOGE(context->GetNodeName(), "Failed to get cacheLineSize."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Slice).Tiling(Tiling4Slice).TilingParse<SliceCompileParam>(TilingPrepare4Slice);
} // namespace optiling