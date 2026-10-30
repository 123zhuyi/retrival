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
 * \file pad_v3_grad_replication_tiling.cpp
 * \brief
 */
#include "pad_v3_grad_replication_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "platform/platform_info.h"

namespace optiling {
static uint32_t GetDataLength32BAligned(uint32_t dataLength, uint32_t dtypeByteSize)
{
    uint32_t eleNumPer32Block = BYTE_32BLOCK / dtypeByteSize;
    return ((dataLength - 1) / eleNumPer32Block + 1) * eleNumPer32Block;
}

template <typename T>
class PadV3GradReplicationTilingHandler {
public:
    explicit PadV3GradReplicationTilingHandler(
        gert::TilingContext* tilingContext, const PadV3GradReplicationCompileInfo* tilingCompileInfo)
        : context(tilingContext), compileInfo(tilingCompileInfo) {};
    ge::graphStatus HandleKernelTiling();
    ge::graphStatus Init();
    void Run();
    void CalculateValues();
    void FillStruct(
        uint64_t layerCount, uint32_t eleCount, uint32_t tileEleNum, EdgeTiling& edgeTiling,
        uint16_t edgeCountLimit = MAX_UINT16);
    void FillStructs();
    void FillValues();
    void SetPreferences();
    void PrintTilingValues();

private:
    gert::TilingContext* context = nullptr;
    PadV3GradReplicationTilingData tilingData;
    const PadV3GradReplicationCompileInfo* compileInfo;
    // 基础信息
    ge::DataType dtype;
    uint32_t dtypeByteSize;
    gert::Shape inputShape;
    size_t inputDim;
    gert::Shape outputShape;
    const T* paddingValue;
    // tiling数据
    uint32_t addTensorBlockNum;
    uint32_t addTensorByteSize;
    uint32_t addTensorSize;
    uint32_t moveTensorBlockNum;
    uint32_t moveTensorByteSize;
    uint32_t moveTensorSize;
    uint32_t virtualInputShape[VIRTUAL_INPUT_DIM];
    uint64_t inputSize;
    uint64_t cubeInputSize;
    uint32_t layerInputSize;
    uint32_t cubeNumEachCore;
    uint32_t realUsedCoreNum;
    uint32_t cubeNumLastCore;
    uint32_t virtualOutputShape[VIRTUAL_INPUT_DIM];
    uint64_t outputSize;
    uint32_t cubeOutputSize;
    uint32_t layerOutputSize;
    uint32_t paddings[PADDING_LENGTH];
    uint32_t topSize;
    uint64_t totalTopInputSizeEachCube;
    int64_t leftSize;
    uint64_t totalLeftInputSizeEachCube;
    int64_t innerRowLength;
    uint32_t topToBottomSize;
    uint64_t topResultSize;
    uint32_t leftToRightSize;
    uint64_t leftResultSize;
    uint64_t workspaceSize;
    uint64_t workspaceByteSize;
};

template <typename T>
ge::graphStatus PadV3GradReplicationTilingHandler<T>::HandleKernelTiling()
{
    ge::graphStatus status = Init();
    if (status)
        return status;
    Run();
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus PadV3GradReplicationTilingHandler<T>::Init()
{
    dtype = context->GetInputTensor(X_INPUT_INDEX)->GetDataType();
    dtypeByteSize = GetSizeByDataType(dtype);
    inputShape = context->GetInputShape(X_INPUT_INDEX)->GetOriginShape();
    inputDim = inputShape.GetDimNum();
    outputShape = context->GetOutputShape(Y_OUTPUT_INDEX)->GetOriginShape();
    auto paddingTensor = context->GetInputTensor(PADDING_INPUT_INDEX);
    paddingValue = paddingTensor->GetData<T>();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "Only support datatype of float32, float16 or bfloat16.");
        return ge::GRAPH_FAILED;
    }
    if (inputDim != DIM_5D && inputDim != DIM_4D) {
        OP_LOGE(context->GetNodeName(), "Only support input of 4 or 5 dimensions.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void PadV3GradReplicationTilingHandler<T>::Run()
{
    CalculateValues();
    FillStructs();
    FillValues();
    SetPreferences();
    PrintTilingValues();
}

/* 计算tiling相关数值 */
template <typename T>
void PadV3GradReplicationTilingHandler<T>::CalculateValues()
{
    if (dtype == ge::DT_FLOAT) {
        addTensorBlockNum = compileInfo->ubByteSize / BYTE_32BLOCK / ADD_TERSOR_NUM;
        addTensorByteSize = addTensorBlockNum * BYTE_32BLOCK;
        addTensorSize = addTensorByteSize / B32_BYTE_SIZE;
        moveTensorBlockNum = compileInfo->ubByteSize / BYTE_32BLOCK;
        moveTensorByteSize = moveTensorBlockNum * BYTE_32BLOCK;
        moveTensorSize = moveTensorByteSize / B32_BYTE_SIZE;
    } else { // fp16、bf16场景，32B对齐后还要cast，因此64B对齐
        addTensorBlockNum = compileInfo->ubByteSize / BYTE_64BLOCK / ADD_TERSOR_NUM;
        addTensorByteSize = addTensorBlockNum * BYTE_64BLOCK;
        addTensorSize = addTensorByteSize / B32_BYTE_SIZE;
        moveTensorBlockNum = compileInfo->ubByteSize / BYTE_64BLOCK;
        moveTensorByteSize = moveTensorBlockNum * BYTE_64BLOCK;
        moveTensorSize = moveTensorByteSize / B32_BYTE_SIZE;
    }

    for (uint32_t i = 1; i < VIRTUAL_INPUT_DIM; i++) {
        virtualInputShape[VIRTUAL_INPUT_DIM - i] = (uint32_t)inputShape[inputDim - i];
    }
    virtualInputShape[DIM_0] = (inputDim == DIM_4D) ? inputShape[DIM_0] : (inputShape[DIM_0] * inputShape[DIM_1]);
    inputSize = context->GetInputTensor(X_INPUT_INDEX)->GetShapeSize();
    cubeInputSize =
        virtualInputShape[DIM_1] * virtualInputShape[DIM_2] * virtualInputShape[DIM_3]; // 计算输入的后三维体积
    layerInputSize = virtualInputShape[DIM_2] * virtualInputShape[DIM_3];               // 计算输入的后两维体积;
    cubeNumEachCore = (virtualInputShape[DIM_0] - 1) / compileInfo->vectorCoreNum + 1;
    realUsedCoreNum = (virtualInputShape[DIM_0] - 1) / cubeNumEachCore + 1;
    cubeNumLastCore = virtualInputShape[DIM_0] % cubeNumEachCore;
    if (cubeNumLastCore == 0)
        cubeNumLastCore = cubeNumEachCore;

    for (uint32_t i = 1; i < VIRTUAL_INPUT_DIM; i++) {
        virtualOutputShape[VIRTUAL_INPUT_DIM - i] = (uint32_t)outputShape[inputDim - i];
    }
    virtualOutputShape[DIM_0] = virtualInputShape[DIM_0];
    outputSize = 1;
    for (uint32_t i = 0; i < outputShape.GetDimNum(); i++) {
        outputSize *= outputShape.GetDim(i);
    }
    cubeOutputSize =
        virtualOutputShape[DIM_1] * virtualOutputShape[DIM_2] * virtualOutputShape[DIM_3]; // 计算输出的后三维体积
    layerOutputSize = virtualOutputShape[DIM_2] * virtualOutputShape[DIM_3];               // 计算输出的后两维体积;
    for (uint32_t i = 0; i < PADDING_PAIR_NUM; i++) {
        paddings[i * PAIR] = paddingValue[PADDING_API_LENGTH - i * PAIR - PAIR];
        paddings[i * PAIR + 1] = paddingValue[PADDING_API_LENGTH - i * PAIR - 1];
    }
    if (virtualOutputShape[DIM_3] == 1) {
        paddings[DIM_0] += paddings[DIM_1];
        paddings[DIM_1] = 0;
    }
    if (virtualOutputShape[DIM_2] == 1) {
        paddings[DIM_2] += paddings[DIM_3];
        paddings[DIM_3] = 0;
    }
    if (virtualOutputShape[DIM_1] == 1) {
        paddings[DIM_4] += paddings[DIM_5];
        paddings[DIM_5] = 0;
    }

    topSize = virtualInputShape[DIM_3];
    totalTopInputSizeEachCube = virtualInputShape[DIM_1] * topSize;
    leftSize = virtualInputShape[DIM_2] - (paddings[DIM_2] ? (paddings[DIM_2] + 1) : 0) -
               (paddings[DIM_3] ? (paddings[DIM_3] + 1) : 0); // 如果小等于0，则没有左右边的中间部分需要累加
    if (leftSize < 0)
        leftSize = 0;
    totalLeftInputSizeEachCube = virtualInputShape[DIM_1] * leftSize;
    innerRowLength = virtualInputShape[DIM_3] - (paddings[DIM_0] ? (paddings[DIM_0] + 1) : 0) -
                     (paddings[DIM_1] ? (paddings[DIM_1] + 1) : 0); // 如果小等于0，则没有中间部分需要搬运
    if (innerRowLength < 0)
        innerRowLength = 0;
    topToBottomSize = (virtualInputShape[DIM_2] - paddings[DIM_3] - 1) * topSize;    // 底边相对顶边在输入上的偏移量
    topResultSize = virtualInputShape[DIM_0] * virtualInputShape[DIM_1] * topSize;   // 底边相对顶边在临时输出上的偏移量
    leftToRightSize = virtualInputShape[DIM_3] - paddings[DIM_1] - 1;                // 右边相对左边在输入上的偏移量
    leftResultSize = virtualInputShape[DIM_0] * virtualInputShape[DIM_1] * leftSize; // 右边相对左边在临时输出上的偏移量
    workspaceSize = (topResultSize + leftResultSize) * PAIR +
                    layerOutputSize * PAIR * virtualInputShape[DIM_0]; // 用户workspace大小
    workspaceByteSize = workspaceSize * B32_BYTE_SIZE;                 // 用户workspace字节占用
}

/* 计算并填写tiling嵌套结构体
   layerCount: 待处理面数
   eleCount: 每面元素数
   tileEleNum: 每轮能处理的元素数
*/
template <typename T>
void PadV3GradReplicationTilingHandler<T>::FillStruct(
    uint64_t layerCount, uint32_t eleCount, uint32_t tileEleNum, EdgeTiling& edgeTiling, uint16_t edgeCountLimit)
{
    uint16_t edgeCount;
    uint64_t tileCount;
    uint16_t additionalCount;
    if (eleCount == 0) {
        edgeTiling.set_edgeCount(0);
        edgeTiling.set_tileCount(0);
        edgeTiling.set_additionalCount(0);
        return;
    }
    if (tileEleNum < eleCount) {
        // large case
        edgeCount = 0;
        tileCount = eleCount / tileEleNum;
        additionalCount = eleCount % tileEleNum;
    } else {
        // small case
        uint32_t fullEleCount = GetDataLength32BAligned(eleCount, dtypeByteSize);
        auto div = tileEleNum / fullEleCount;
        if (div <= edgeCountLimit) {
            edgeCount = div;
        } else {
            edgeCount = edgeCountLimit;
        }
        tileCount = layerCount / edgeCount;
        additionalCount = layerCount % edgeCount;
    }
    edgeTiling.set_edgeCount(edgeCount);
    edgeTiling.set_tileCount(tileCount);
    edgeTiling.set_additionalCount(additionalCount);
}

/* 填写所有tiling嵌套结构体 */
template <typename T>
void PadV3GradReplicationTilingHandler<T>::FillStructs()
{
    uint64_t layerCount;
    uint32_t eleCount;
    uint32_t tileEleNum;
    // topTiling
    layerCount = cubeNumEachCore * virtualInputShape[DIM_1];
    eleCount = topSize;
    tileEleNum = addTensorSize;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.topTiling);
    // leftTiling
    eleCount = leftSize;
    tileEleNum = addTensorBlockNum;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.leftTiling);
    // cornerTiling
    layerCount = cubeNumEachCore * virtualInputShape[DIM_1];
    tilingData.cornerTiling.set_edgeCount(0);
    tilingData.cornerTiling.set_tileCount(layerCount / tileEleNum);
    tilingData.cornerTiling.set_additionalCount(layerCount % tileEleNum);
    // innerTiling
    layerCount = leftSize;
    eleCount = innerRowLength;
    tileEleNum = moveTensorSize;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.innerTiling, PARAM_LIMIT_4095);
    // paddingLayerTiling
    layerCount = cubeNumEachCore;
    eleCount = layerOutputSize;
    tileEleNum = moveTensorSize;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.paddingLayerTiling, PARAM_LIMIT_4095);

    // topTilingLastCore
    layerCount = cubeNumLastCore * virtualInputShape[DIM_1];
    eleCount = topSize;
    tileEleNum = addTensorSize;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.topTilingLastCore);
    // leftTilingLastCore
    eleCount = leftSize;
    tileEleNum = addTensorBlockNum;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.leftTilingLastCore);
    // cornerTilingLastCore
    layerCount = cubeNumLastCore * virtualInputShape[DIM_1];
    tilingData.cornerTilingLastCore.set_edgeCount(0);
    tilingData.cornerTilingLastCore.set_tileCount(layerCount / tileEleNum);
    tilingData.cornerTilingLastCore.set_additionalCount(layerCount % tileEleNum);
    // paddingLayerTilingLastCore
    layerCount = cubeNumLastCore;
    eleCount = layerOutputSize;
    tileEleNum = moveTensorSize;
    FillStruct(layerCount, eleCount, tileEleNum, tilingData.paddingLayerTilingLastCore, PARAM_LIMIT_4095);
}

/* 填写tiling结构体数值 */
template <typename T>
void PadV3GradReplicationTilingHandler<T>::FillValues()
{
    tilingData.set_addTensorBlockNum(addTensorBlockNum);
    tilingData.set_addTensorByteSize(addTensorByteSize);
    tilingData.set_addTensorSize(addTensorSize);
    tilingData.set_moveTensorBlockNum(moveTensorBlockNum);
    tilingData.set_moveTensorByteSize(moveTensorByteSize);
    tilingData.set_moveTensorSize(moveTensorSize);
    tilingData.set_inputShape(virtualInputShape);
    tilingData.set_inputSize(inputSize);
    tilingData.set_cubeInputSize(cubeInputSize);
    tilingData.set_layerInputSize(layerInputSize);
    tilingData.set_cubeNumEachCore(cubeNumEachCore);
    tilingData.set_realUsedCoreNum(realUsedCoreNum);
    tilingData.set_cubeNumLastCore(cubeNumLastCore);
    tilingData.set_outputShape(virtualOutputShape);
    tilingData.set_outputSize(outputSize);
    tilingData.set_cubeOutputSize(cubeOutputSize);
    tilingData.set_layerOutputSize(layerOutputSize);
    tilingData.set_paddings(paddings);
    tilingData.set_topSize(topSize);
    tilingData.set_totalTopInputSizeEachCube(totalTopInputSizeEachCube);
    tilingData.set_leftSize(leftSize);
    tilingData.set_totalLeftInputSizeEachCube(totalLeftInputSizeEachCube);
    tilingData.set_innerRowLength(innerRowLength);
    tilingData.set_topToBottomSize(topToBottomSize);
    tilingData.set_topResultSize(topResultSize);
    tilingData.set_leftToRightSize(leftToRightSize);
    tilingData.set_leftResultSize(leftResultSize);
    tilingData.set_workspaceSize(workspaceSize);
}

static void PrintEdgeTiling(const char* name, EdgeTiling& tiling, const ge::char_t* nodeName)
{
    OP_LOGD(nodeName, "%s: {", name);
    OP_LOGD(nodeName, "  edgeCount: %u", tiling.get_edgeCount());
    OP_LOGD(nodeName, "  tileCount: %lu", tiling.get_tileCount());
    OP_LOGD(nodeName, "  additionalCount: %u", tiling.get_additionalCount());
    OP_LOGD(nodeName, "}");
}

static void PrintPadV3GradReplicationTilingData(PadV3GradReplicationTilingData& data, const ge::char_t* nodeName)
{
    PrintEdgeTiling("topTiling", data.topTiling, nodeName);
    PrintEdgeTiling("leftTiling", data.leftTiling, nodeName);
    PrintEdgeTiling("cornerTiling", data.cornerTiling, nodeName);
    PrintEdgeTiling("innerTiling", data.innerTiling, nodeName);
    PrintEdgeTiling("paddingLayerTiling", data.paddingLayerTiling, nodeName);
    PrintEdgeTiling("topTilingLastCore", data.topTilingLastCore, nodeName);
    PrintEdgeTiling("leftTilingLastCore", data.leftTilingLastCore, nodeName);
    PrintEdgeTiling("cornerTilingLastCore", data.cornerTilingLastCore, nodeName);
    PrintEdgeTiling("paddingLayerTilingLastCore", data.paddingLayerTilingLastCore, nodeName);
}

template <typename T>
void PadV3GradReplicationTilingHandler<T>::PrintTilingValues()
{
    const ge::char_t* nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "addTensorBlockNum: %u", addTensorBlockNum);
    OP_LOGD(nodeName, "addTensorByteSize: %u", addTensorByteSize);
    OP_LOGD(nodeName, "addTensorSize: %u", addTensorSize);
    OP_LOGD(nodeName, "moveTensorBlockNum: %u", moveTensorBlockNum);
    OP_LOGD(nodeName, "moveTensorByteSize: %u", moveTensorByteSize);
    OP_LOGD(nodeName, "moveTensorSize: %u", moveTensorSize);
    OP_LOGD(nodeName, "inputShape[0]: %u", virtualInputShape[DIM_0]);
    OP_LOGD(nodeName, "inputShape[1]: %u", virtualInputShape[DIM_1]);
    OP_LOGD(nodeName, "inputShape[2]: %u", virtualInputShape[DIM_2]);
    OP_LOGD(nodeName, "inputShape[3]: %u", virtualInputShape[DIM_3]);
    OP_LOGD(nodeName, "inputSize: %lu", inputSize);
    OP_LOGD(nodeName, "cubeInputSize: %lu", cubeInputSize);
    OP_LOGD(nodeName, "layerInputSize: %u", layerInputSize);
    OP_LOGD(nodeName, "cubeNumEachCore: %u", cubeNumEachCore);
    OP_LOGD(nodeName, "realUsedCoreNum: %u", realUsedCoreNum);
    OP_LOGD(nodeName, "cubeNumLastCore: %u", cubeNumLastCore);
    OP_LOGD(nodeName, "outputShape[0]: %u", virtualOutputShape[DIM_0]);
    OP_LOGD(nodeName, "outputShape[1]: %u", virtualOutputShape[DIM_1]);
    OP_LOGD(nodeName, "outputShape[2]: %u", virtualOutputShape[DIM_2]);
    OP_LOGD(nodeName, "outputShape[3]: %u", virtualOutputShape[DIM_3]);
    OP_LOGD(nodeName, "outputSize: %lu", outputSize);
    OP_LOGD(nodeName, "cubeOutputSize: %u", cubeOutputSize);
    OP_LOGD(nodeName, "layerOutputSize: %u", layerOutputSize);
    OP_LOGD(nodeName, "paddings[0]: %u", paddings[DIM_0]);
    OP_LOGD(nodeName, "paddings[1]: %u", paddings[DIM_1]);
    OP_LOGD(nodeName, "paddings[2]: %u", paddings[DIM_2]);
    OP_LOGD(nodeName, "paddings[3]: %u", paddings[DIM_3]);
    OP_LOGD(nodeName, "paddings[4]: %u", paddings[DIM_4]);
    OP_LOGD(nodeName, "paddings[5]: %u", paddings[DIM_5]);
    OP_LOGD(nodeName, "topSize: %u", topSize);
    OP_LOGD(nodeName, "totalTopInputSizeEachCube: %lu", totalTopInputSizeEachCube);
    OP_LOGD(nodeName, "leftSize: %ld", leftSize);
    OP_LOGD(nodeName, "totalLeftInputSizeEachCube: %lu", totalLeftInputSizeEachCube);
    OP_LOGD(nodeName, "innerRowLength: %ld", innerRowLength);
    OP_LOGD(nodeName, "topToBottomSize: %u", topToBottomSize);
    OP_LOGD(nodeName, "topResultSize: %lu", topResultSize);
    OP_LOGD(nodeName, "leftToRightSize: %u", leftToRightSize);
    OP_LOGD(nodeName, "leftResultSize: %lu", leftResultSize);
    OP_LOGD(nodeName, "workspaceSize: %lu", workspaceSize);
    PrintPadV3GradReplicationTilingData(tilingData, nodeName);
}

/* 配置tiling */
template <typename T>
void PadV3GradReplicationTilingHandler<T>::SetPreferences()
{
    // 设置tiling key、block dim
    context->SetBlockDim(realUsedCoreNum);
    if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(REPLICATION_FP32_KEY);
    } else if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(REPLICATION_FP16_KEY);
    } else if (dtype == ge::DT_BF16) {
        context->SetTilingKey(REPLICATION_BF16_KEY);
    }
    // 设置workspace大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);                    // 通过框架获取workspace的指针
    currentWorkspace[0] = workspaceByteSize + compileInfo->sysWorkspaceByteSize; // 设置总的workspace的数值大小
    // 写入tiling
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetScheduleMode(1); // SyncAll need set
}

static ge::graphStatus PadV3GradReplicationTiling(gert::TilingContext* context)
{
    const PadV3GradReplicationCompileInfo* compileInfo =
        reinterpret_cast<const PadV3GradReplicationCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGI(context->GetNodeName(), "Calculating tiling for PadV3GradReplication.");
    const gert::Tensor* paddings_tensor = context->GetInputTensor(PADDING_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddings_tensor);
    ge::DataType paddings_dtype = paddings_tensor->GetDataType();
    if (paddings_dtype == ge::DT_INT32) {
        PadV3GradReplicationTilingHandler<int32_t> handler(context, compileInfo);
        return handler.HandleKernelTiling();
    } else if (paddings_dtype == ge::DT_INT64) {
        PadV3GradReplicationTilingHandler<int64_t> handler(context, compileInfo);
        return handler.HandleKernelTiling();
    } else {
        OP_LOGE(context->GetNodeName(), "Only support padding value in INT64 or INT32.");
        return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus PadV3GradReplicationTilingParse(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<PadV3GradReplicationCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 填充AscendC的compile info
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "No vector core available."), return ge::GRAPH_FAILED);
    uint64_t ubByteSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubByteSize);
    compileInfo->ubByteSize = ubByteSize;
    OP_CHECK_IF(
        (compileInfo->ubByteSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceByteSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
// register tiling interface of the PadV3GradReplication op.
IMPL_OP_OPTILING(PadV3GradReplication)
    .Tiling(PadV3GradReplicationTiling)
    .TilingParse<PadV3GradReplicationCompileInfo>(PadV3GradReplicationTilingParse)
    .TilingInputsDataDependency({PADDING_INPUT_INDEX});
}; // namespace optiling