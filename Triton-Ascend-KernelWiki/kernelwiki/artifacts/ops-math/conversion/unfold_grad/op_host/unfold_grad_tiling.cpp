/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unfold_grad_tiling.cpp
 * \brief
 */

#include "unfold_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"

namespace {
constexpr uint32_t MAX_DIM_NUM = 8;
constexpr uint32_t INPUT_GRADOUT_IDX = 0;
constexpr uint32_t INPUT_INPUTSIZE_IDX = 1;
constexpr uint32_t FP16BF16_TYPESIZE = 2;
constexpr uint32_t FP32_TYPESIZE = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t ATTR_0 = 0;
constexpr uint64_t ATTR_1 = 1;
constexpr uint64_t ATTR_2 = 2;
constexpr int32_t TRANS_BLOCK = 16;
constexpr int32_t ONCE_TRANSDATATO5HD_SIZE = 512;
constexpr int32_t DOUBLE = 2;
constexpr int32_t TOTAL_SIZE = 5;
constexpr int32_t FP16BF16_SIZE_CEIL = 73;
constexpr int32_t FP32_SIZE_CEIL = 89;

constexpr int32_t DATA_TYPE = 10;
constexpr int32_t OVERLAPPED = 200;
constexpr int32_t NONOVERLAPPING = 300;
constexpr int32_t FINALAXE = 1;
constexpr int32_t FINALSECONDAXE = 2;
constexpr int32_t TYPE_MODE1 = 1; // 为fp32
constexpr int32_t TYPE_MODE2 = 2; // 为fp16
constexpr int32_t TYPE_MODE3 = 3; // 为bf16
constexpr uint32_t ONE_OPERATION_TQUE_NUM = 2;
constexpr uint32_t ONE_OPERATION_TQUE_SIZE = 17;
constexpr uint32_t BIG_SIZE = 1000;
constexpr uint32_t SIZE_LEVEL_NORMAL = 0; // under the limit of UB Trans
constexpr uint32_t SIZE_LEVEL_BIG = 1; // beyond the limit of UB Trans
} // namespace

namespace optiling {
using namespace std;

template <typename T>
ge::graphStatus CopyData2Array(const gert::Tensor* listTensor, int64_t listSize, int64_t dataList[])
{
    const T* listDataPtr = listTensor->GetData<T>();
    if (listDataPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < listSize; i++) {
        dataList[i] = listDataPtr[i];
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus GetConstInputData(
    gert::TilingContext* context, const size_t idxInput, T& dataList, int64_t& dataListLength)
{
    auto listTensor = context->GetInputTensor(idxInput);
    if (listTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inputSizeDesc = context->GetInputDesc(INPUT_INPUTSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeDesc);
    auto listDataType = inputSizeDesc->GetDataType();
    auto inputSizeShapePtr = context->GetInputShape(INPUT_INPUTSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeShapePtr);
    auto inputSizeShape = inputSizeShapePtr->GetStorageShape();
    int64_t listSize = static_cast<int64_t>(inputSizeShape.GetDim(0));
    dataListLength = listSize;

    if (listDataType == ge::DT_INT32) {
        return CopyData2Array<int32_t>(listTensor, listSize, dataList);
    }
    if (listDataType == ge::DT_INT64) {
        return CopyData2Array<int64_t>(listTensor, listSize, dataList);
    }

    return ge::GRAPH_FAILED;
}

void UnfoldGradTiling::PrintInfo()
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "----------------Start to print UnfoldGradTiling tiling data.----------------");
    OP_LOGD(nodeName, "batchNum = %ld.", tiling.get_batchNum());
    OP_LOGD(nodeName, "batchNumPerCore = %ld.", tiling.get_batchNumPerCore());
    OP_LOGD(nodeName, "batchNumTailCore = %ld.", tiling.get_batchNumTailCore());
    OP_LOGD(nodeName, "maxBatchNum4Ub = %ld.", tiling.get_maxBatchNum4Ub());
    OP_LOGD(nodeName, "useCoreNum = %ld.", tiling.get_useCoreNum());
    OP_LOGD(nodeName, "ubSizeT1 = %ld.", tiling.get_ubSizeT1());
    OP_LOGD(nodeName, "ubSizeT2 = %ld.", tiling.get_ubSizeT2());
    OP_LOGD(nodeName, "outputNumPerCore = %ld.", tiling.get_outputNumPerCore());
    OP_LOGD(nodeName, "inputNumPerCore = %ld.", tiling.get_inputNumPerCore());
    OP_LOGD(nodeName, "iterationNumPerCore = %ld.", tiling.get_iterationNumPerCore());
    OP_LOGD(nodeName, "handleNUMOnceIterationPerCore = %ld.", tiling.get_handleNUMOnceIterationPerCore());
    OP_LOGD(nodeName, "tasksOnceMaxPerCore = %ld.", tiling.get_tasksOnceMaxPerCore());
    OP_LOGD(nodeName, "inputSizeLength = %ld.", tiling.get_inputSizeLength());
    OP_LOGD(nodeName, "rowAvailableLengthSrc = %ld.", tiling.get_rowAvailableLengthSrc());
    OP_LOGD(nodeName, "lowestCommonMultiple = %ld.", tiling.get_lowestCommonMultiple());
    OP_LOGD(nodeName, "colOnceMaxPerUB = %ld.", tiling.get_colOnceMaxPerUB());
    OP_LOGD(nodeName, "tailColLength = %ld.", tiling.get_tailColLength());
    OP_LOGD(nodeName, "typeSizeT1 = %ld.", tiling.get_typeSizeT1());
    OP_LOGD(nodeName, "typeSizeT2 = %ld.", tiling.get_typeSizeT2());
    OP_LOGD(nodeName, "width = %ld.", tiling.get_width());
    OP_LOGD(nodeName, "gradOutSizeDim = %ld.", tiling.get_gradOutSizeDim());
    OP_LOGD(nodeName, "inputSizeLastDim = %ld.", tiling.get_inputSizeLastDim());
    OP_LOGD(nodeName, "dim = %ld.", tiling.get_dim());
    OP_LOGD(nodeName, "size = %ld.", tiling.get_size());
    OP_LOGD(nodeName, "step = %ld.", tiling.get_step());
    OP_LOGD(nodeName, "loop = %ld.", tiling.get_loop());
    OP_LOGD(nodeName, "tail = %ld.", tiling.get_tail());
    OP_LOGD(nodeName, "sizeLevel = %ld.", tiling.get_sizeLevel());
    OP_LOGD(nodeName, "----------------print UnfoldGradTiling tiling data end.----------------");
}

void UnfoldGradTiling::SetTilingData()
{
    context->SetBlockDim(useCoreNum);
    tiling.set_batchNum(batchNum);
    tiling.set_batchNumPerCore(batchNumPerCore);
    tiling.set_batchNumTailCore(batchNumTailCore);
    tiling.set_maxBatchNum4Ub(maxBatchNum4Ub);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_ubSizeT2(ubSizeT2);
    tiling.set_ubSizeT1(ubSizeT1);
    tiling.set_outputNumPerCore(outputNumPerCore);
    tiling.set_inputNumPerCore(inputNumPerCore);
    tiling.set_iterationNumPerCore(iterationNumPerCore);
    tiling.set_handleNUMOnceIterationPerCore(handleNUMOnceIterationPerCore);
    tiling.set_tasksOnceMaxPerCore(tasksOnceMaxPerCore);
    tiling.set_inputSizeLength(inputSizeLength);
    tiling.set_rowAvailableLengthSrc(rowAvailableLengthSrc);
    tiling.set_lowestCommonMultiple(lowestCommonMultiple);
    tiling.set_colOnceMaxPerUB(colOnceMaxPerUB);
    tiling.set_tailColLength(tailColLength);
    tiling.set_typeSizeT1(typeSizeT1);
    tiling.set_typeSizeT2(typeSizeT2);
    tiling.set_width(width);
    tiling.set_gradOutSizeDim(gradOutSizeDim);
    tiling.set_dim(dim);
    tiling.set_inputSizeLastDim(inputSizeLastDim);
    tiling.set_size(size);
    tiling.set_step(step);
    tiling.set_loop(loop);
    tiling.set_tail(tail);
    tiling.set_sizeLevel(sizeLevel);
}

ge::graphStatus UnfoldGradTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    totalCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = static_cast<int64_t>(ubSizePlatForm);

    return ge::GRAPH_SUCCESS;
}

bool UnfoldGradTiling::GetDataTypeKey(ge::DataType dataType)
{
    width = 8; // transdataTo5HD的行数，即一次处理的有效数据量。当前一律转为fp32计算，故该值为8

    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            typeSizeT1 = FP16BF16_TYPESIZE;
            dataTypeTilingKey = (dataType == ge::DT_FLOAT16 ? TYPE_MODE2 : TYPE_MODE3);
            break;
        case ge::DT_FLOAT:
            typeSizeT1 = FP32_TYPESIZE;
            dataTypeTilingKey = TYPE_MODE1;
            break;
        default:
            return false;
    }

    return true;
}

ge::graphStatus UnfoldGradTiling::GetInputTensorInfo()
{
    auto nodeName = context->GetNodeName();

    // 获取第二个输入inputSize
    int64_t inputSizeArray[MAX_DIM_NUM] = {0};

    if (GetConstInputData(context, INPUT_INPUTSIZE_IDX, inputSizeArray, inputSizeLength) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetConstInputData inputSize failed.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(size <= 0, OP_LOGE(nodeName, "Size %ld must be greater than 0.", size), return ge::GRAPH_FAILED);
    OP_CHECK_IF(step <= 0, OP_LOGE(nodeName, "Step %ld must be greater than 0.", size), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dim < 0 || dim >= inputSizeLength,
        OP_LOGE(nodeName, "Dim %ld must be greater than or equal to 0 and less than the shape size of inputSize.", dim),
        return ge::GRAPH_FAILED);

    // 获取第一个输入gradOut的信息
    auto gradOutShapePtr = context->GetInputShape(INPUT_GRADOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutShapePtr);
    auto gradOutShape = gradOutShapePtr->GetStorageShape();

    OP_CHECK_IF(
        gradOutShape.GetDimNum() != static_cast<size_t>(inputSizeLength) + 1,
        OP_LOGE(
            nodeName, "The number of dimension in the gradOut %ld must be equal to (inputSizeLength + 1) %ld.",
            static_cast<int64_t>(gradOutShape.GetDimNum()), static_cast<int64_t>(inputSizeLength) + 1),
        return ge::GRAPH_FAILED);
    gradOutSizeDim = gradOutShape[dim];
    OP_CHECK_IF(
        gradOutSizeDim != (inputSizeArray[dim] - size) / step + 1,
        OP_LOGE(
            nodeName, "The size of dimension dim in the gradOut %ld must be ((inputSize[dim] - size) / step + 1) %ld.",
            gradOutSizeDim, (inputSizeArray[dim] - size) / step + 1),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gradOutShape[gradOutShape.GetDimNum() - 1] != size,
        OP_LOGE(
            nodeName, "The last dimension of gradOut %ld must be size %ld.", gradOutShape[gradOutShape.GetDimNum() - 1],
            size),
        return ge::GRAPH_FAILED);

    inputSizeLastDim = inputSizeArray[inputSizeLength - 1];
    for (size_t i = dim; i < static_cast<size_t>(inputSizeLength); i++) {
        outputNumPerCore *= inputSizeArray[i];
    }
    for (size_t i = 0; i < static_cast<size_t>(inputSizeLength); i++) {
        usrWorkspaceSize *= inputSizeArray[i];
    }
    usrWorkspaceSize *= FP32_TYPESIZE;
    for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
        batchNum *= gradOutShape[i];
    }
    for (size_t i = dim; i < gradOutShape.GetDimNum(); ++i) {
        inputNumPerCore *= gradOutShape[i];
    }

    auto gradOutDesc = context->GetInputDesc(INPUT_GRADOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutDesc);
    gardInDType = gradOutDesc->GetDataType();
    OP_CHECK_IF(
        GetDataTypeKey(gardInDType) == false,
        OP_LOGE(nodeName, "The dtype of input gradOut must be in [float32, float16, bfloat16]."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnfoldGradTiling::TilingBlock4SizeLevelBig()
{
    auto nodeName = context->GetNodeName();

    handleNUMOnceIterationPerCore = gradOutSizeDim * size;
    // 计算size对齐32占用空间
    int64_t neededSpaceForOnePaddedSize = 0;
    
    // 重新划分UB空间
    if (gardInDType == ge::DT_FLOAT16 || gardInDType == ge::DT_BF16) {
        ubSizeT2 = ubSize / (TOTAL_SIZE - DOUBLE) * DOUBLE;
        ubSizeT1 = ubSize / (TOTAL_SIZE - DOUBLE);
        neededSpaceForOnePaddedSize = (size * typeSizeT1 + BLOCK_SIZE - 1) 
                                            / BLOCK_SIZE * BLOCK_SIZE
                                            / typeSizeT1 * FP32_TYPESIZE;
    } else if (gardInDType == ge::DT_FLOAT) {
        ubSizeT2 = ubSize;
        neededSpaceForOnePaddedSize = (size * FP32_TYPESIZE + BLOCK_SIZE - 1) 
                                            / BLOCK_SIZE * BLOCK_SIZE;
    } else {
        return ge::GRAPH_FAILED;
    }
    
    tasksOnceMaxPerCore = ubSizeT2 / neededSpaceForOnePaddedSize * size;

    OP_CHECK_IF(
        tasksOnceMaxPerCore <= 0,
        OP_LOGE(nodeName, "TasksOnceMaxPerCore %ld must be greater than 0. This is because size should be less than %ld.", tasksOnceMaxPerCore, ubSizeT2 / FP32_TYPESIZE),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnfoldGradTiling::Tiling4Block()
{
    auto nodeName = context->GetNodeName();

    if (gardInDType == ge::DT_FLOAT16 || gardInDType == ge::DT_BF16) {
        ubSizeT2 = ubSize / (ONCE_TRANSDATATO5HD_SIZE * TOTAL_SIZE) * (ONCE_TRANSDATATO5HD_SIZE * DOUBLE);
        ubSizeT1 = ubSize / (ONCE_TRANSDATATO5HD_SIZE * TOTAL_SIZE) * ONCE_TRANSDATATO5HD_SIZE;
    } else if (gardInDType == ge::DT_FLOAT) {
        ubSizeT2 = ubSize / DOUBLE / (ONCE_TRANSDATATO5HD_SIZE * DOUBLE) * (ONCE_TRANSDATATO5HD_SIZE * DOUBLE);
    } else {
        return ge::GRAPH_FAILED;
    }

    // 分核计算
    useCoreNum = static_cast<int64_t>(Ops::Base::CeilDiv(batchNum, Ops::Base::CeilDiv(batchNum, totalCoreNum)));
    batchNumPerCore = (batchNum + useCoreNum - 1) / useCoreNum;
    batchNumTailCore = batchNum - (useCoreNum - 1) * batchNumPerCore;
    // 分块计算
    int64_t neededAmountForOneSize = size;
    if (dim == inputSizeLength - 1 && step >= size) {
        neededAmountForOneSize = step;
    }

    if (dim == inputSizeLength - 1) {
        // 计算ub使用Trans的上限
        int64_t colHandleNum = ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) /neededAmountForOneSize * size;
        // 判断size是否超出
        sizeLevel = neededAmountForOneSize > colHandleNum ? SIZE_LEVEL_BIG : SIZE_LEVEL_NORMAL;
        // 最后一轴的情况下：对BigSize进行处理
        if (sizeLevel == SIZE_LEVEL_BIG){
            OP_CHECK_IF(
                TilingBlock4SizeLevelBig() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "TilingBlock4SizeLevelBig failed."),
                return ge::GRAPH_FAILED);
        }else{
            handleNUMOnceIterationPerCore = gradOutSizeDim * size;
            tasksOnceMaxPerCore = ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / neededAmountForOneSize * size * TRANS_BLOCK; 
        }
    } else {
        iterationNumPerCore = gradOutSizeDim;
        handleNUMOnceIterationPerCore = inputSizeLastDim;
        rowAvailableLengthSrc = (size + width - 1) / width * width;
        lowestCommonMultiple = getLowestCommonMultiple(size, TRANS_BLOCK);
        int64_t rowT2NeededLength = lowestCommonMultiple / size * rowAvailableLengthSrc;
        colOnceMaxPerUB = ubSizeT2 / FP32_TYPESIZE / rowT2NeededLength / TRANS_BLOCK * TRANS_BLOCK;
        tasksOnceMaxPerCore = colOnceMaxPerUB * lowestCommonMultiple / neededAmountForOneSize;

        int32_t maxSize = FP16BF16_SIZE_CEIL;
        if(typeSizeT1 == FP32_TYPESIZE) {
            maxSize = FP32_SIZE_CEIL;
        }
        OP_CHECK_IF(
            tasksOnceMaxPerCore <= 0,
            OP_LOGE(nodeName, "TasksOnceMaxPerCore %ld must be greater than 0. This is because size should be less than %d.", tasksOnceMaxPerCore, maxSize),
            return ge::GRAPH_FAILED);
    }

    loop = handleNUMOnceIterationPerCore / tasksOnceMaxPerCore;
    tail = handleNUMOnceIterationPerCore % tasksOnceMaxPerCore;
    OP_CHECK_IF(
        lowestCommonMultiple <= 0,
        OP_LOGE(nodeName, "lowestCommonMultiple %ld must be greater than 0.", lowestCommonMultiple),
        return ge::GRAPH_FAILED);
    tailColLength = (tail * neededAmountForOneSize + lowestCommonMultiple - 1) /
                    lowestCommonMultiple; // tailColLength仅用于非尾轴情况

    OP_CHECK_IF(
        inputNumPerCore <= 0 && outputNumPerCore <= 0,
        OP_LOGE(
            nodeName, "InputNumPerCore %ld or outputNumPerCore %ld must be greater than 0.", inputNumPerCore,
            outputNumPerCore),
        return ge::GRAPH_FAILED);
    maxBatchNum4Ub = tasksOnceMaxPerCore / Max(inputNumPerCore, outputNumPerCore);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnfoldGradTiling::SetAttrParams()
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(ATTR_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimPtr);
    dim = static_cast<int64_t>(*dimPtr);
    const int64_t* sizePtr = attrs->GetAttrPointer<int64_t>(ATTR_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizePtr);
    size = static_cast<int64_t>(*sizePtr);
    const int64_t* stepPtr = attrs->GetAttrPointer<int64_t>(ATTR_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stepPtr);
    step = static_cast<int64_t>(*stepPtr);

    tiling.set_dim(dim);
    tiling.set_size(size);
    tiling.set_step(step);

    return ge::GRAPH_SUCCESS;
}

void UnfoldGradTiling::SetTilingKey()
{
    if (step < size) {
        tilingKey_ = OVERLAPPED + DATA_TYPE * dataTypeTilingKey;
    } else {
        tilingKey_ = NONOVERLAPPING + DATA_TYPE * dataTypeTilingKey;
    }
    if (dim == inputSizeLength - 1) {
        if (sizeLevel == SIZE_LEVEL_BIG) {
            tilingKey_ += BIG_SIZE;
        }
        tilingKey_ += FINALAXE;
    } else {
        tilingKey_ += FINALSECONDAXE;
    }
}

uint64_t UnfoldGradTiling::GetTilingKey()
{
    return tilingKey_;
}

ge::graphStatus UnfoldGradTiling::DoTiling()
{
    auto nodeName = context->GetNodeName();

    OP_CHECK_IF(
        SetAttrParams() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "SetAttrParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetInputTensorInfo failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        Tiling4Block() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling4Block failed."), return ge::GRAPH_FAILED);
    SetTilingData();

    SetTilingKey();
    context->SetTilingKey(GetTilingKey());

    PrintInfo();
    // 固定写法
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] =
        usrWorkspaceSize +
        ascendcPlatform.GetLibApiWorkSpaceSize(); // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
                                                  // 该workspace作为中转空间用来放T2类型数据

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UnfoldGrad(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling4UnfoldGrad running begin.");

    UnfoldGradTiling tilingObj(context);

    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4UnfoldGrad(gert::TilingParseContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling4UnfoldGrad running end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UnfoldGrad)
    .Tiling(Tiling4UnfoldGrad)
    .TilingParse<Tiling4UnfoldGradCompileInfo>(TilingPrepare4UnfoldGrad)
    .TilingInputsDataDependency({1});
} // namespace optiling