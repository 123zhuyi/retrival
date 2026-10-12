/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file dynamic_stitch_tiling_arch35.cpp
 * \brief
 */
#include "dynamic_stitch_tiling_arch35.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_api/op_util.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_class.h"

using namespace Ops::Math::OpTiling;
using namespace Ops::Base;

namespace optiling {
constexpr int64_t INPUT_INDICES_IDX = 0;
constexpr int64_t INPUT_X_IDX = 1;
constexpr int64_t ATTR_N_IDX = 0;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int32_t MAX_TENSOR_NUM = 64;
constexpr int64_t DYNAMIC_STITCH_TILING_PRIORITY = 10000;
constexpr uint64_t TILING_KEY_FOR_SIMT = 100000;
constexpr uint64_t TILING_KEY_FOR_SIMD = 200000;
constexpr int32_t MIN_SMID_SLICE_VALUE = 256;
constexpr size_t RESERVED_WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int64_t SIMT_DCACHE_SIZE = static_cast<int64_t>(64 * 1024);
constexpr int64_t MIN_INDICES_BUFFER_SIZE = 32 * sizeof(int32_t);

static const std::vector<ge::DataType> X_SUPPORT_DTYPE = {
    ge::DT_INT8,   ge::DT_UINT8, ge::DT_INT16,   ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_BOOL,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64};

ge::graphStatus DynamicStitchTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const DynamicStitchCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->blockDim;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::CheckAndGetParam()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto indiceInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indiceInstanceInfo);
    auto xInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInstanceInfo);
    totalTensorCnt_ = indiceInstanceInfo->GetInstanceNum();
    OP_CHECK_IF(
        totalTensorCnt_ > MAX_TENSOR_NUM,
        OP_LOGE_FOR_INVALID_TENSORNUM(
            context_->GetNodeName(), "indices", totalTensorCnt_, std::to_string(MAX_TENSOR_NUM).c_str()),
        return ge::GRAPH_FAILED);
    if (totalTensorCnt_ != static_cast<int64_t>(xInstanceInfo->GetInstanceNum())) {
        std::string listLenMsg =
            std::to_string(totalTensorCnt_) + " and " + std::to_string(xInstanceInfo->GetInstanceNum());
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            context_->GetNodeName(), "indices and x", listLenMsg.c_str(),
            "The number of tensors in indices and x must be the same");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        CheckAndGetIndiceInputList() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckAndGetIndiceInputList failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckAndGetXInputList() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckAndGetXInputList failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckAndGetOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckAndGetOutput failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Check attr failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::CheckAndGetIndiceInputList()
{
    totalTensorSum_ = 0;
    for (int32_t i = 0; i < totalTensorCnt_; i++) {
        auto indicesDesc = context_->GetDynamicInputDesc(INPUT_INDICES_IDX, i);
        OP_CHECK_IF(
            indicesDesc == nullptr, OP_LOGE(context_->GetNodeName(), "The input indices[%d]'s desc is null.", i),
            return ge::GRAPH_FAILED);
        auto currShape = context_->GetDynamicInputShape(INPUT_INDICES_IDX, i);
        OP_CHECK_IF(
            currShape == nullptr, OP_LOGE(context_->GetNodeName(), "The input indices[%d]'s shape is null.", i),
            return ge::GRAPH_FAILED);
        if (indicesDesc->GetDataType() != ge::DT_INT32) {
            std::string paramMsg = "indices " + std::to_string(i) + "th tensor";
            OP_LOGE_FOR_INVALID_DTYPE(
                context_->GetNodeName(), paramMsg.c_str(), Ops::Base::ToString(indicesDesc->GetDataType()).c_str(),
                "int32");
            return ge::GRAPH_FAILED;
        }

        if (CheckShapeAllNonNeg(currShape->GetStorageShape()) != ge::GRAPH_SUCCESS) {
            std::string paramMsg = "indices " + std::to_string(i) + "th tensor";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), paramMsg.c_str(), Ops::Base::ToString(currShape->GetStorageShape()).c_str(),
                "The input indices's tensor has negative dimension");
            return ge::GRAPH_FAILED;
        }
        auto currShapeSize = currShape->GetStorageShape().GetShapeSize();
        OP_LOGI(context_->GetNodeName(), "Indices[%d]'s shape size is %ld", i, currShapeSize);
        tensorCumsumList_[i] = totalTensorSum_;
        tensorCntList_[i] = currShapeSize;
        totalTensorSum_ += currShapeSize;
        OP_LOGI(context_->GetNodeName(), "Indices[%d]'s cumsum shape is %ld.", i, totalTensorSum_);
    }
    tensorCumsumList_[totalTensorCnt_] = totalTensorSum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::CheckAndGetXInputList()
{
    dataType_ = ge::DT_UNDEFINED;
    for (int32_t i = 0; i < totalTensorCnt_; i++) {
        auto tempDesc = context_->GetDynamicInputDesc(INPUT_X_IDX, i);
        OP_CHECK_IF(
            tempDesc == nullptr, OP_LOGE(context_->GetNodeName(), "The input x[%d]'s desc is null.", i),
            return ge::GRAPH_FAILED);
        auto currShape = context_->GetDynamicInputShape(INPUT_X_IDX, i);
        OP_CHECK_IF(
            currShape == nullptr, OP_LOGE(context_->GetNodeName(), "The input x[%d]'s shape is null.", i),
            return ge::GRAPH_FAILED);
        auto srcDtype = tempDesc->GetDataType();
        if (std::find(X_SUPPORT_DTYPE.begin(), X_SUPPORT_DTYPE.end(), srcDtype) == X_SUPPORT_DTYPE.end()) {
            std::string paramMsg = "x " + std::to_string(i) + "th tensor";
            OP_LOGE_FOR_INVALID_DTYPE(
                context_->GetNodeName(), paramMsg.c_str(), Ops::Base::ToString(srcDtype).c_str(),
                "INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, FLOAT16, BF16, FLOAT, DOUBLE and "
                "COMPLEX64");
            return ge::GRAPH_FAILED;
        }
        if (dataType_ == ge::DT_UNDEFINED) {
            dataType_ = srcDtype;
        } else if (srcDtype != dataType_) {
            std::string paramMsg = "x " + std::to_string(i) + "th tensor";
            std::string reasonMsg = "The dtypes of all tensors in x must be the same, but dtype of the " + std::to_string(i) +
                                    "th tensor is inconsistent with other dtypes " + Ops::Base::ToString(dataType_);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), paramMsg.c_str(), Ops::Base::ToString(srcDtype).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (CheckShapeAllNonNeg(currShape->GetStorageShape()) != ge::GRAPH_SUCCESS) {
            std::string paramMsg = "x " + std::to_string(i) + "th tensor";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), paramMsg.c_str(), Ops::Base::ToString(currShape->GetStorageShape()).c_str(),
                "The input x's tensor has negative or empty dimension");
            return ge::GRAPH_FAILED;
        }
    }
    return CheckAndGetSliceSize();
}

ge::graphStatus DynamicStitchTilingClass::CheckAndGetSliceSize()
{
    for (int32_t i = 0; i < totalTensorCnt_; i++) {
        auto& indicesShape = context_->GetDynamicInputShape(INPUT_INDICES_IDX, i)->GetStorageShape();
        auto& xShape = context_->GetDynamicInputShape(INPUT_X_IDX, i)->GetStorageShape();
        auto indiceDimNum = indicesShape.GetDimNum();
        auto xDimNum = xShape.GetDimNum();
        if (indiceDimNum > xDimNum) {
            std::string paramMsg = "indices " + std::to_string(i) + "th tensor and " + "x " + std::to_string(i) + "th tensor";
            std::string dimMsg = std::to_string(indiceDimNum) + " and " + std::to_string(xDimNum);
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context_->GetNodeName(), paramMsg.c_str(), dimMsg.c_str(),
                "The shape dimension of indice's tensor must be less than or equal to the dimension of x's corresponding tensor");
            return ge::GRAPH_FAILED;
        }
        for (size_t j = 0; j < indiceDimNum; j++) {
            if (indicesShape.GetDim(j) != xShape.GetDim(j)) {
                std::string paramMsg = "indices " + std::to_string(i) + "th tensor and " + "x " + std::to_string(i) + "th tensor";
                std::string shapeMsg = Ops::Base::ToString(indicesShape) + " and " + Ops::Base::ToString(xShape);
                std::string reasonMsg = "The shape of indices's tensor should match the shape formed by the first " +
                                        std::to_string(indiceDimNum) + " axes of x's corresponding tensor";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), shapeMsg.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        if (i == 0) {
            sliceShape_ = GetSliceShapeFromIndiceAndXShape(indicesShape, xShape);
            sliceSize_ = 1;
            for (size_t index = 0; index < sliceShape_.size(); index++) {
                sliceSize_ *= sliceShape_[index];
            }
        } else {
            auto currSliceShape = GetSliceShapeFromIndiceAndXShape(indicesShape, xShape);
            if (!IsTwoSliceShapeEqual(sliceShape_, currSliceShape)) {
                std::string paramMsg = "indices " + std::to_string(i) + "th tensor and " + "x " + std::to_string(i) + "th tensor";
                std::string shapeMsg = Ops::Base::ToString(indicesShape) + " and " + Ops::Base::ToString(xShape);
                std::string reasonMsg =
                    "All x[i].shape - indices[i].shape must be the same, actually x[0].shape - indices[0].shape is " +
                    ops::ToStringWithSize(sliceShape_.data(), sliceShape_.size()) + ", x[" + std::to_string(i) +
                    "].shape - indices[" + std::to_string(i) + "].shape is " +
                    ops::ToStringWithSize(currSliceShape.data(), currSliceShape.size());
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), shapeMsg.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::CheckAndGetOutput()
{
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_IF(
        outputDesc == nullptr, OP_LOGE(context_->GetNodeName(), "output's desc is nullptr."),
        return ge::GRAPH_FAILED);
    auto outputDtype = outputDesc->GetDataType();
    if (outputDtype != dataType_) {
        std::string reasonMsg = "The dtype of output y must be the same as the dtypes " +
                                Ops::Base::ToString(dataType_) + " of all input x's tensors";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(outputDtype).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        context_->GetOutputShape(OUTPUT_IDX) == nullptr, OP_LOGE(context_->GetNodeName(), "output's shape is nullptr."),
        return ge::GRAPH_FAILED);
    auto& outputShape = context_->GetOutputShape(OUTPUT_IDX)->GetStorageShape();
    OP_CHECK_IF(
        CheckShapeAllNonNeg(outputShape) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(outputShape).c_str(),
            "The output y has negative or empty axes"),
        return ge::GRAPH_FAILED);
    auto outputDimNum = outputShape.GetDimNum();
    std::vector<int64_t> outputSlice;
    for (size_t i = 1; i < outputDimNum; i++) {
        outputSlice.emplace_back(outputShape.GetDim(i));
    }
    if (!IsTwoSliceShapeEqual(outputSlice, sliceShape_)) {
        std::string reasonMsg =
            "All x[i].shape - indices[i].shape must be the same as output.shape[1:-1], actually x[0].shape - "
            "indices[0].shape is " +
            ops::ToStringWithSize(sliceShape_.data(), sliceShape_.size()) + ", output.shape[1:-1] is " +
            ops::ToStringWithSize(outputSlice.data(), outputSlice.size());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(outputShape).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    maxIndex_ = outputShape.GetDim(0) - 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::CheckAttr() const
{
    // attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto attrNPtr = attrs->GetInt(ATTR_N_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrNPtr);
    int32_t attrNValue = static_cast<int32_t>(*attrNPtr);
    OP_CHECK_IF(
        attrNValue < 1,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "N", std::to_string(attrNValue).c_str(),
            "greater than or equal to 1"),
        return ge::GRAPH_FAILED);
    if (attrNValue != totalTensorCnt_) {
        std::string reasonMsg = "The value of attr N must be equal to actual tensor count " + std::to_string(totalTensorCnt_);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "N", std::to_string(attrNValue).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::DoOpTiling()
{
    auto ret = CheckAndGetParam();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    int64_t indiceAlignFactor = GetUbBlockSize(context_) / static_cast<int32_t>(sizeof(int32_t));
    // 计算writeBack分核
    auto writeBackBlockSizeUnAlign = CeilDiv(maxIndex_ + 1, static_cast<int64_t>(aicoreParams_.numBlocks));
    writeBackBlockSize_ = CeilAlign(writeBackBlockSizeUnAlign, indiceAlignFactor);
    writeBackBlockNum_ = CeilDiv(maxIndex_ + 1, writeBackBlockSize_);
    writeBackTailBlockSize_ = ((maxIndex_ + 1) % writeBackBlockSize_) == 0 ?
                                  writeBackBlockSize_ :
                                  ((maxIndex_ + 1) % writeBackBlockSize_);
    // 计算clearWorkspace分核
    int64_t workSpaceSize = maxIndex_ + 1 + totalTensorSum_;
    int64_t clrBlockWsSizeUnAlign = CeilDiv(workSpaceSize, static_cast<int64_t>(aicoreParams_.numBlocks));
    clrBlockWsSize_ = CeilAlign(clrBlockWsSizeUnAlign, indiceAlignFactor);
    clrBlockNum_ = CeilDiv(workSpaceSize, clrBlockWsSize_);
    clrTailBlockWsSize_ = (workSpaceSize % clrBlockWsSize_) == 0 ?
                              clrBlockWsSize_ :
                              (workSpaceSize % clrBlockWsSize_);

    int64_t blockDim = std::min(static_cast<uint32_t>(aicoreParams_.numBlocks), MAX_CORE_CONT);
    auto blockFactor = CeilDiv(totalTensorSum_, blockDim);
    usedCoreNum_ = CeilDiv(totalTensorSum_, blockFactor);
    blockFactor_ = CeilDiv(totalTensorSum_, usedCoreNum_);
    tailBlockFactor_ = (totalTensorSum_ % blockFactor_) == 0 ? blockFactor_ : (totalTensorSum_ % blockFactor_);
    AssignDataToEachCore();
    ClassifySliceType();
    // total usable ub size with double buffer
    int64_t ubSize = (aicoreParams_.ubSize - SIMT_DCACHE_SIZE) / 2;
    int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context_));
    int64_t ubAlign = FloorAlign(ubSize, ubBlockSize);
    int64_t sliceLen = sliceSize_ * static_cast<int64_t>(sliceType_);
    if (MIN_INDICES_BUFFER_SIZE + CeilAlign(sliceLen, ubBlockSize) > ubAlign) {
        indicesBufferSize_ = MIN_INDICES_BUFFER_SIZE;
        ubFactor_ = (ubAlign - indicesBufferSize_) / static_cast<int32_t>(sliceType_);
        ubLoopTimes_ = CeilDiv(sliceSize_, static_cast<int64_t>(ubFactor_));
        ubTailFactor_ = sliceSize_ % ubFactor_ == 0 ? ubFactor_ : sliceSize_ % ubFactor_;
    } else {
        ubFactor_ = sliceSize_;
        ubLoopTimes_ = 1;
        ubTailFactor_ = sliceSize_;
        indicesBufferSize_ = ubAlign - CeilAlign(sliceLen, ubBlockSize);
    }
    indicesBufferSize_ =
        std::min(indicesBufferSize_, CeilAlign(static_cast<int64_t>(blockFactor_ * sizeof(int32_t)), ubBlockSize));
    return ge::GRAPH_SUCCESS;
}

void DynamicStitchTilingClass::AssignDataToEachCore()
{
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t currCoreBorder = 0;
    int64_t curDataPos = 0;
    tensorStartList_[coreIndex] = 0;
    tensorStartOffsetList_[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCnt_; i++) {
        if (coreIndex < usedCoreNum_) {
            currCoreBorder = blockFactor_;
        } else {
            currCoreBorder = tailBlockFactor_;
        }
        int64_t tempCount = tensorCntList_[i] - curDataPos;

        // 当前Tensor全部在当前core上处理
        if (dataCount + tempCount < currCoreBorder) {
            dataCount += tempCount;
            curDataPos = 0;
            continue;
        }
        // dataCount >= currCoreBorder, Calculate the offset
        tensorEndList_[coreIndex] = i;
        curDataPos = curDataPos + currCoreBorder - dataCount;
        tensorEndOffsetList_[coreIndex] = curDataPos - 1;
        // 重新初始化dataCount,开始计算下一个core的dataCount
        dataCount = 0;
        coreIndex++;
        if (curDataPos < tensorCntList_[i]) {
            tensorStartList_[coreIndex] = i;
            tensorStartOffsetList_[coreIndex] = curDataPos;
            --i; // The next loop continues to allocate the current tensor
        } else if (coreIndex != usedCoreNum_) {
            tensorStartList_[coreIndex] = i + 1;
            tensorStartOffsetList_[coreIndex] = 0;
            curDataPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount > 0) {
        tensorEndList_[coreIndex] = totalTensorCnt_ - 1;
        tensorEndOffsetList_[coreIndex] = tensorCntList_[totalTensorCnt_ - 1] - 1;
    }
}

void DynamicStitchTilingClass::ClassifySliceType()
{
    auto sliceLen = sliceSize_ * ge::GetSizeByDataType(dataType_);
    if (sliceLen % static_cast<int64_t>(SliceDivisorType::SLICE_EIGHT) == 0) {
        sliceType_ = SliceDivisorType::SLICE_EIGHT;
        sliceSize_ = sliceLen / static_cast<int64_t>(SliceDivisorType::SLICE_EIGHT);
    } else if (sliceLen % static_cast<int64_t>(SliceDivisorType::SLICE_FOUR) == 0) {
        sliceType_ = SliceDivisorType::SLICE_FOUR;
        sliceSize_ = sliceLen / static_cast<int64_t>(SliceDivisorType::SLICE_FOUR);
    } else if (sliceLen % static_cast<int64_t>(SliceDivisorType::SLICE_TWO) == 0) {
        sliceType_ = SliceDivisorType::SLICE_TWO;
        sliceSize_ = sliceLen / static_cast<int64_t>(SliceDivisorType::SLICE_TWO);
    } else {
        sliceType_ = SliceDivisorType::SLICE_ONE;
        sliceSize_ = sliceLen;
    }
}

ge::graphStatus DynamicStitchTilingClass::GetWorkspaceSize()
{
    auto workSpaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaces);
    workSpaces[0] = RESERVED_WORKSPACE_SIZE;
    auto usrWorkSpaceSize = (totalTensorSum_ + maxIndex_ + 1) * sizeof(int32_t);
    workSpaces[0] = workSpaces[0] + usrWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicStitchTilingClass::PostTiling()
{
    tilingData_ = context_->GetTilingData<DynamicStitchTilingData>();
    tilingData_->sliceType = static_cast<int64_t>(sliceType_);
    tilingData_->sliceSize = sliceSize_;
    tilingData_->clrBlockNum = clrBlockNum_;
    tilingData_->clrBlockWsSize = clrBlockWsSize_;
    tilingData_->clrTailBlockWsSize = clrTailBlockWsSize_;
    tilingData_->writeBackBlockNum = writeBackBlockNum_;
    tilingData_->writeBackBlockSize = writeBackBlockSize_;
    tilingData_->writeBackTailBlockSize = writeBackTailBlockSize_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->tailBlockFactor = tailBlockFactor_;
    tilingData_->indicesBufferSize = indicesBufferSize_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->ubTailFactor = ubTailFactor_;
    tilingData_->ubLoopTimes = ubLoopTimes_;
    tilingData_->totalTensorSum = totalTensorSum_;
    tilingData_->totalTensorCnt = totalTensorCnt_;
    tilingData_->maxIndex = maxIndex_;
    for (size_t i = 0; i < MAX_CORE_CONT; i++) {
        tilingData_->tensorStartList[i] = tensorStartList_[i];
        tilingData_->tensorEndList[i] = tensorEndList_[i];
        tilingData_->tensorStartOffsetList[i] = tensorStartOffsetList_[i];
        tilingData_->tensorEndOffsetList[i] = tensorEndOffsetList_[i];
    }
    for (size_t i = 0; i <= MAX_LIST_TENSOR_CNT; i++) {
        tilingData_->tensorCumsumList[i] = tensorCumsumList_[i];
    }
    if (maxIndex_ <= -1 || sliceSize_ <= 0) {
        context_->SetBlockDim(1);
        tilingData_->clrBlockNum = 0;
        tilingData_->usedCoreNum = 0;
        tilingData_->writeBackBlockNum = 0;
    } else {
        context_->SetBlockDim(std::max(std::max(clrBlockNum_, usedCoreNum_), writeBackBlockNum_));
    }
    context_->SetTilingKey(GetTilingKey());
    context_->SetLocalMemorySize(aicoreParams_.ubSize - SIMT_DCACHE_SIZE);
    context_->SetScheduleMode(1);
    PrintTiling();
    return ge::GRAPH_SUCCESS;
}

bool DynamicStitchTilingClass::IsBigSliceSize() const
{
    return sliceSize_ >= MIN_SMID_SLICE_VALUE;
}

void DynamicStitchTilingClass::PrintTiling() const
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is sliceType: %ld, sliceSize: %ld, clrBlockNum: %ld, clrBlockWsSize: %ld,"
        "clrTailBlockWsSize: %ld, writeBackBlockNum: %ld, writeBackBlockSize: %ld, writeBackTailBlockSize: %ld, "
        "usedCoreNum: %ld, blockFactor: %ld, tailBlockFactor: %ld, indicesBufferSize: %ld, ubFactor: %ld, "
        "ubTailFactor: %ld, ubLoopTimes: %ld, totalTensorSum: %ld, totalTensorCnt: %ld, maxIndex: %ld, TilingKey: %ld",
        tilingData_->sliceType, tilingData_->sliceSize, tilingData_->clrBlockNum, tilingData_->clrBlockWsSize,
        tilingData_->clrTailBlockWsSize, tilingData_->writeBackBlockNum, tilingData_->writeBackBlockSize,
        tilingData_->writeBackTailBlockSize, tilingData_->usedCoreNum, tilingData_->blockFactor,
        tilingData_->tailBlockFactor, tilingData_->indicesBufferSize, tilingData_->ubFactor, tilingData_->ubTailFactor,
        tilingData_->ubLoopTimes, tilingData_->totalTensorSum, tilingData_->totalTensorCnt, tilingData_->maxIndex,
        GetTilingKey());
    for (int64_t i = 0; i < usedCoreNum_; i++) {
        OP_LOGI(
            context_->GetNodeName(),
            "tensorStartList[%ld] = %hu, tensorEndList[%ld] = %hu, tensorStartOffsetList[%ld] = %ld, "
            "tensorEndOffsetList[%ld] = %ld.",
            i, tilingData_->tensorStartList[i], i, tilingData_->tensorEndList[i], i,
            tilingData_->tensorStartOffsetList[i], i, tilingData_->tensorEndOffsetList[i]);
    }
    for (int64_t i = 0; i <= totalTensorCnt_; i++) {
        OP_LOGI(context_->GetNodeName(), "tensorCumsumList[%ld] = %ld", i, tilingData_->tensorCumsumList[i]);
    }
    return;
}

ge::graphStatus DynamicStitchTilingClass::CheckShapeAllNonNeg(const gert::Shape& shape) const
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) < 0) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t DynamicStitchTilingClass::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (IsBigSliceSize()) {
        tilingKey = TILING_KEY_FOR_SIMD;
    } else {
        tilingKey = TILING_KEY_FOR_SIMT;
    }
    return tilingKey + static_cast<uint64_t>(sliceType_);
}

std::vector<int64_t> DynamicStitchTilingClass::GetSliceShapeFromIndiceAndXShape(
    const gert::Shape& indiceShape, const gert::Shape& xShape) const
{
    std::vector<int64_t> sliceShape;
    if (indiceShape.GetDimNum() == xShape.GetDimNum()) {
        return sliceShape;
    }
    for (size_t i = indiceShape.GetDimNum(); i < xShape.GetDimNum(); i++) {
        sliceShape.emplace_back(xShape.GetDim(i));
    }
    return sliceShape;
}

bool DynamicStitchTilingClass::IsTwoSliceShapeEqual(
    const std::vector<int64_t>& sliceShape1, const std::vector<int64_t>& sliceShape2) const
{
    if (sliceShape1.size() != sliceShape2.size()) {
        return false;
    }
    for (size_t i = 0; i < sliceShape1.size(); i++) {
        if (sliceShape1[i] != sliceShape2[i]) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus Tiling4DynamicStitchTiling(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4DynamicStitchTiling(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetCompiledInfo<DynamicStitchCompileInfo>());
    auto platformInfoPtr = context->GetPlatformInfo();
    auto compileInfoPtr = context->GetCompiledInfo<DynamicStitchCompileInfo>();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->blockDim = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicStitch)
    .Tiling(Tiling4DynamicStitchTiling)
    .TilingParse<DynamicStitchCompileInfo>(TilingPrepare4DynamicStitchTiling);

REGISTER_OPS_TILING_TEMPLATE(DynamicStitch, DynamicStitchTilingClass, DYNAMIC_STITCH_TILING_PRIORITY);
} // namespace optiling