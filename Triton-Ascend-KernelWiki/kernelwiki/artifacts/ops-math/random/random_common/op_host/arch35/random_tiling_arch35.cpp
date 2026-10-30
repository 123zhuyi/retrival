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
 * \file random_tiling_arch35.cpp
 * \brief
 */

#include <algorithm>
#include "platform/platform_infos_def.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "random_tiling_base.h"
#include "random_tiling_arch35.h"
#include <cstdlib>

namespace optiling {
static constexpr int64_t MIN_CORE_PRO = 256;
static constexpr int32_t SPLIT_PUSH_COUNT = 2;

int64_t TensorSliceState::GetMaxOffsetBytes() const
{
    int64_t maxOffsetBytes = 1L;
    for (int64_t dim = 0; dim < ndim; dim++) {
        maxOffsetBytes += (shape[dim] - 1) * strides[dim] * elementSize;
    }
    return maxOffsetBytes;
}

bool TensorSliceState::Is32bitIndexable() const
{
    int64_t maxValue = static_cast<int64_t>(INDEX_32BIT_LIMIT);
    if (numel > maxValue)
        return false;
    if (GetMaxOffsetBytes() > maxValue)
        return false;
    return true;
}

int64_t TensorSliceState::GetDimToSplit() const
{
    int64_t maxExtent = -1, dimToSplit = -1;
    for (int64_t dim = ndim - 1; dim >= 0; dim--) {
        if (shape[dim] == 0)
            continue;
        int64_t extent = (shape[dim] - 1) * std::abs(strides[dim]) * elementSize;
        if (extent > maxExtent) {
            maxExtent = extent;
            dimToSplit = dim;
        }
    }
    return dimToSplit;
}

void TensorSliceState::ReduceDimExtent(int64_t dim, int64_t start, int64_t size)
{
    gmOffset += start * strides[dim];
    shape[dim] = size;
    numel = 1;
    for (int64_t d = 0; d < ndim; d++)
        numel *= shape[d];
}

void TensorSliceState::PartitionDim(int64_t dim, TensorSliceState& other)
{
    int64_t copySize = shape[dim] / 2;
    int64_t thisSize = shape[dim] - copySize;
    for (int64_t d = 0; d < ndim; d++) {
        other.shape[d] = shape[d];
        other.strides[d] = strides[d];
    }
    other.ndim = ndim;
    other.elementSize = elementSize;
    other.gmOffset = gmOffset;
    other.ReduceDimExtent(dim, 0, copySize);
    ReduceDimExtent(dim, copySize, thisSize);
}

ge::graphStatus InitTensorSliceState(TensorSliceState& state, const gert::Shape& outputTensor, int64_t outputSize,
                                     ge::DataType outputDtype)
{
    state.numel = outputSize;
    state.gmOffset = 0;
    state.elementSize = ge::GetSizeByDataType(outputDtype);

    int64_t rawShape[MAX_TENSOR_DIMS] = {0};
    int64_t rawStrides[MAX_TENSOR_DIMS] = {0};
    int64_t rawNdim = static_cast<int64_t>(outputTensor.GetDimNum());
    for (int64_t dim = 0; dim < rawNdim && dim < MAX_TENSOR_DIMS; dim++) {
        rawShape[dim] = outputTensor.GetDim(static_cast<size_t>(dim));
    }
    if (rawNdim > 0) {
        rawStrides[rawNdim - 1] = 1;
        for (int64_t dim = rawNdim - 2; dim >= 0; dim--) {
            rawStrides[dim] = rawShape[dim + 1] * rawStrides[dim + 1];
        }
    }

    int64_t coalescedShape[MAX_TENSOR_DIMS] = {0};
    int64_t coalescedStrides[MAX_TENSOR_DIMS] = {0};
    int64_t coalescedNdim = 0;
    for (int64_t dim = 0; dim < rawNdim; dim++) {
        if (rawShape[dim] == 1) {
            continue;
        }
        coalescedShape[coalescedNdim] = rawShape[dim];
        coalescedStrides[coalescedNdim] = rawStrides[dim];
        coalescedNdim++;
    }
    if (coalescedNdim == 0) {
        coalescedShape[0] = 1;
        coalescedStrides[0] = 1;
        coalescedNdim = 1;
    }
    for (int64_t dim = coalescedNdim - 1; dim > 0; dim--) {
        if (coalescedShape[dim] * coalescedStrides[dim] == coalescedStrides[dim - 1]) {
            coalescedShape[dim - 1] *= coalescedShape[dim];
            coalescedStrides[dim - 1] = coalescedStrides[dim];
            for (int64_t d = dim; d < coalescedNdim - 1; d++) {
                coalescedShape[d] = coalescedShape[d + 1];
                coalescedStrides[d] = coalescedStrides[d + 1];
            }
            coalescedNdim--;
        }
    }

    state.ndim = coalescedNdim;
    for (int64_t dim = 0; dim < coalescedNdim && dim < MAX_TENSOR_DIMS; dim++) {
        state.shape[dim] = coalescedShape[dim];
        state.strides[dim] = coalescedStrides[dim];
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcSplitBlocks(TensorSliceState& state, RandomUnifiedSimtTilingDataStruct& simtTilingData)
{
    state.gmOffset = 0;
    simtTilingData.splitBlockCount = 0;

    if (state.Is32bitIndexable()) {
        simtTilingData.splitBlocks[0].numel = state.numel;
        simtTilingData.splitBlocks[0].gmOffset = state.gmOffset;
        simtTilingData.splitBlockCount = 1;
    } else {
        TensorSliceState stack[MAX_SPLIT_BLOCKS];
        int32_t top = 0;
        stack[top++] = state;

        while (top > 0 && simtTilingData.splitBlockCount < MAX_SPLIT_BLOCKS) {
            TensorSliceState cur = stack[--top];

            if (cur.Is32bitIndexable()) {
                int64_t blockIdx = simtTilingData.splitBlockCount;
                simtTilingData.splitBlocks[blockIdx].numel = cur.numel;
                simtTilingData.splitBlocks[blockIdx].gmOffset = cur.gmOffset;
                simtTilingData.splitBlockCount++;
            } else {
                int64_t dim = cur.GetDimToSplit();
                TensorSliceState other;
                cur.PartitionDim(dim, other);
                if (static_cast<int32_t>(MAX_SPLIT_BLOCKS) - top < SPLIT_PUSH_COUNT) {
                    return ge::GRAPH_FAILED;
                }
                stack[top++] = cur;
                stack[top++] = other;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcExecutionPoliciesForBlocks(RandomUnifiedSimtTilingDataStruct& simtTilingData, uint32_t unrollFactor)
{
    int64_t accumulatedOffset = simtTilingData.offset;

    if (simtTilingData.splitBlockCount > 1) {
        int64_t totalNumel = simtTilingData.outputSize;
        int64_t grid = (totalNumel + SIMT_THREAD_GROUP_SIZE - 1) / SIMT_THREAD_GROUP_SIZE;
        grid = std::max(grid, 1L); // 空tensor场景grid至少为1，避免除0
        int64_t blocksPerAic = MAX_THREADS_PER_AIC / SIMT_THREAD_GROUP_SIZE;
        grid = (AIC_CLUSTER_COUNT * blocksPerAic < grid) ? AIC_CLUSTER_COUNT * blocksPerAic : grid;

        int64_t totalThreads = grid * SIMT_THREAD_GROUP_SIZE;
        int64_t threadsPerRound = totalThreads * unrollFactor;
        int64_t roundsNeeded = (totalNumel + threadsPerRound - 1) / threadsPerRound;
        int64_t counterOffset = roundsNeeded * MAX_PRNG_COUNTER_INCR;

        accumulatedOffset += counterOffset;
    }

    for (int64_t i = 0; i < simtTilingData.splitBlockCount; i++) {
        int64_t numel = simtTilingData.splitBlocks[i].numel;

        int64_t grid = (numel + SIMT_THREAD_GROUP_SIZE - 1) / SIMT_THREAD_GROUP_SIZE;
        grid = std::max(grid, 1L); // 空tensor场景grid至少为1，避免除0
        int64_t blocksPerAic = MAX_THREADS_PER_AIC / SIMT_THREAD_GROUP_SIZE;
        grid = (AIC_CLUSTER_COUNT * blocksPerAic < grid) ? AIC_CLUSTER_COUNT * blocksPerAic : grid;

        int64_t totalThreads = grid * SIMT_THREAD_GROUP_SIZE;
        int64_t threadsPerRound = totalThreads * unrollFactor;
        int64_t roundsNeeded = (numel + threadsPerRound - 1) / threadsPerRound;
        int64_t counterOffset = roundsNeeded * MAX_PRNG_COUNTER_INCR;

        simtTilingData.splitBlocks[i].kernelOffset = accumulatedOffset;
        simtTilingData.splitBlocks[i].grid = grid;
        simtTilingData.splitBlocks[i].totalThreads = totalThreads;

        accumulatedOffset += counterOffset;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingParseArch35(gert::TilingParseContext* context, const std::string& operatorName)
{
    OP_LOGD(context, "Entering RandomTilingArch35  operator name : %s", operatorName);
    auto compileInfo = context->GetCompiledInfo<RandomOperatorCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context, "GetHardwareInfo Failed, vectorCoreNum:%ld, ubSize:%ld.", compileInfo->totalCoreNum,
                        compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static inline ge::graphStatus GetIntValue(const gert::TilingContext* context, const gert::Tensor* constTensor,
                                          gert::Shape& constShape)
{
    const T* constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, constValue);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExtractTensorValue(const gert::TilingContext* context, const int64_t constIdx, gert::Shape& constShape)
{
    auto constTensor = context->GetRequiredInputTensor(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, constTensor);

    auto inputDescPtr = context->GetRequiredInputDesc(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    auto constDtype = inputDescPtr->GetDataType();

    auto ret = ge::GRAPH_FAILED;
    switch (constDtype) {
        case ge::DT_INT32:
            ret = GetIntValue<int32_t>(context, constTensor, constShape);
            break;
        case ge::DT_INT64:
            ret = GetIntValue<int64_t>(context, constTensor, constShape);
            break;
        default:
            OP_LOGD(context->GetNodeName(), "ExtractTensorValue only support [int32, int64]. but is %s",
                    Ops::Base::ToString(constDtype).c_str());
            return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "get const value failed, please check."),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "current const value is %s", Ops::Base::ToString(constShape).c_str());
    return ge::GRAPH_SUCCESS;
}

RandomTilingArch35::RandomTilingArch35(gert::TilingContext* context, const OpTilingConfig& config)
    : context_(context), config_(config), tilingData_{}, simtTilingData_{}
{}

ge::graphStatus RandomTilingArch35::DoTiling()
{
    opName_ = context_->GetNodeName();
    OP_LOGD(opName_, "Start tiling for op: %s", opName_.c_str());
    // 步骤1：校验输入输出和属性
    auto ret = CheckInputsOutputsAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Check inputs/outputs/attrs failed");
        return ret;
    }

    // 步骤2： 获取硬件信息
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Get platform info failed");
        return ret;
    }

    // 步骤3：前置处理（可选）
    ret = BeforeProcess();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Before process  failed");
        return ret;
    }

    // 步骤4： 填充TilingData
    ret = config_.kernelMode == RandomKernelMode::SIMD ? FillUnifiedTilingData() : FillUnifiedSimtTilingData();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fill tiling data failed");
        return ret;
    }

    // 步骤5：计算tilingKey和workspace
    ret = CalcTilingKeyAndWorkspace();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Calc tiling key/workspace failed");
        return ret;
    }

    // 步骤6：后置处理（可选）
    ret = UniqueProcess();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Unique process  failed");
        return ret;
    }

    // 步骤7：写入context
    ret = WriteBackToContext();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Write tiling data to context failed");
        return ret;
    }

    // 步骤8：调用dump函数
    OP_LOGI(
        "RandomTiling", "%s",
        (config_.kernelMode == RandomKernelMode::SIMD ? tilingData_.DumpTilingInfo() : simtTilingData_.DumpTilingInfo())
            .c_str());
    OP_LOGD(opName_, "Tiling success for op: %s", opName_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::CheckInputsOutputsAndAttrs()
{
    OP_LOGI(opName_, "TilingContext: %s", RandomUtils::GetTilingContext(context_).c_str());
    for (const auto& [idx, rule] : config_.inputCheckRules) {
        auto tensorDesc = context_->GetRequiredInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, tensorDesc);
        auto inputShape = context_->GetRequiredInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        auto storageShape = inputShape->GetStorageShape();

        auto ret = CheckTensor(tensorDesc, storageShape, rule, "input_" + std::to_string(idx));
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    for (const auto& [idx, rule] : config_.optionalInputCheckRules) {
        auto tensorDesc = context_->GetOptionalInputDesc(idx);
        auto inputShape = context_->GetOptionalInputShape(idx);
        if (tensorDesc != nullptr && inputShape != nullptr) {
            auto storageShape = inputShape->GetStorageShape();
            auto ret = CheckTensor(tensorDesc, storageShape, rule, "optional_input_" + std::to_string(idx));
            if (ret != ge::GRAPH_SUCCESS) {
                return ret;
            }
        }
    }

    for (const auto& [idx, rule] : config_.outputCheckRules) {
        auto tensorDesc = context_->GetOutputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, tensorDesc);
        auto outputShape = context_->GetOutputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
        auto storageShape = outputShape->GetStorageShape();

        auto ret = CheckTensor(tensorDesc, storageShape, rule, "output_" + std::to_string(idx));
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    // 校验属性
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    for (const auto& rule : config_.attrCheckRules) {
        auto attrIdx = rule.first;
        auto checkFunc = rule.second;
        if (!checkFunc(context_)) {
            OP_LOGE(context_->GetNodeName(), "Attr %d check failed", attrIdx);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = static_cast<const RandomOperatorCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        auto aivNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF((aivNum <= 0), OP_LOGE(opName_, "RandomTilingArch35 fail to get coreNum."),
                    return ge::GRAPH_FAILED);
        totalCoreNum_ = aivNum;
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    ubSize_ -= config_.DcacheSize;

    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size less than Dcache Size. please check."),
                return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomTilingArch35::GetPlatformInfo ubSize_=%d, totalCoreNum_=%d", ubSize_, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::DoSimtBlockTiling()
{
    OP_CHECK_IF((totalCoreNum_ <= 0), OP_LOGE(opName_, "totalCoreNum is less than or equal to 0. please check."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((config_.coreAlignSize == 0), OP_LOGE(opName_, "coreAlignSize is  equal to 0. please check."),
                return ge::GRAPH_FAILED);

    int64_t avgPerCore = Ops::Base::CeilDiv(simtTilingData_.outputSize, totalCoreNum_);
    int64_t numOfPerCore = Ops::Base::CeilAlign(avgPerCore, config_.coreAlignSize);
    int64_t usedCoreNum = Ops::Base::CeilDiv(simtTilingData_.outputSize, numOfPerCore);
    simtTilingData_.usedCoreNum = std::max(1L, std::min(totalCoreNum_, usedCoreNum)); // 空tensor至少开1核
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::FillUnifiedSimtTilingData()
{
    auto ret = config_.getOutputSize(context_, simtTilingData_.outputSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_CHECK_IF((simtTilingData_.outputSize < 0), OP_LOGE(opName_, "outputSize is less than 0. please check."),
                return ge::GRAPH_FAILED);
    ret = config_.getSeedAndOffset(context_, simtTilingData_.seed, simtTilingData_.offset);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoSimtBlockTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (config_.enableSplitBlocks) {
        auto outputShape = context_->GetOutputShape(config_.splitOutputIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
        auto outputTensor = outputShape->GetStorageShape();
        auto outputDesc = context_->GetOutputDesc(config_.splitOutputIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
        auto outputDtype = outputDesc->GetDataType();

        TensorSliceState state;
        InitTensorSliceState(state, outputTensor, simtTilingData_.outputSize, outputDtype);

        ret = CalcSplitBlocks(state, simtTilingData_);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        ret = CalcExecutionPoliciesForBlocks(simtTilingData_, config_.unrollFactor);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::FillUnifiedTilingData()
{
    // 1. 调用算子回调函数
    auto ret = config_.getOutputSize(context_, tilingData_.outputSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_CHECK_IF((tilingData_.outputSize <= 0), OP_LOGE(opName_, "outputSize is less than or equal to 0. please check."),
                return ge::GRAPH_FAILED);
    ret = config_.getKeyAndCounter(context_, tilingData_.key, tilingData_.counter);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = config_.getBufferNum(context_, bufNum_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 2. 通用分核计算
    ret = DoBlockTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 3. 通用分UB计算
    ret = DoUbTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 4. 填充其他字段
    tilingData_.sharedTmpBufSize = config_.sharedTmpBufSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::DoBlockTiling()
{
    // 通用分核逻辑：按outputSize均分
    OP_CHECK_IF((totalCoreNum_ <= 0), OP_LOGE(opName_, "totalCoreNum is less than or equal to 0. please check."),
                return ge::GRAPH_FAILED);
    tilingData_.normalCoreProNum = Ops::Base::CeilDiv(tilingData_.outputSize, totalCoreNum_);
    OP_CHECK_IF((config_.coreAlignSize == 0), OP_LOGE(opName_, "coreAlignSize is  equal to 0. please check."),
                return ge::GRAPH_FAILED);
    tilingData_.normalCoreProNum = (tilingData_.normalCoreProNum + config_.coreAlignSize - 1) / config_.coreAlignSize *
                                   config_.coreAlignSize;
    tilingData_.normalCoreProNum = std::max(tilingData_.normalCoreProNum, MIN_CORE_PRO);
    tilingData_.usedCoreNum = Ops::Base::CeilDiv(tilingData_.outputSize, tilingData_.normalCoreProNum);
    tilingData_.usedCoreNum = std::max(1L, tilingData_.usedCoreNum); // 空tensor至少开1核
    tilingData_.tailCoreProNum = tilingData_.outputSize - tilingData_.normalCoreProNum * (tilingData_.usedCoreNum - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::DoUbTiling()
{
    ubSize_ -= config_.sharedTmpBufSize;
    // 通用分UB逻辑：根据存活节点数量计算 singleBufferSize
    OP_CHECK_IF((bufNum_ == 0), OP_LOGE(opName_, "bufNum_ is equal to 0. please check."), return ge::GRAPH_FAILED);
    tilingData_.singleBufferSize = ubSize_ / bufNum_;
    if (config_.ubAlignSize == 0) {
        auto ubBlockSize = Ops::Base::GetUbBlockSize(context_);
        OP_CHECK_IF((ubBlockSize == 0), OP_LOGE(opName_, "ubBlockSize is equal to 0. please check."),
                    return ge::GRAPH_FAILED);
        tilingData_.singleBufferSize = tilingData_.singleBufferSize / ubBlockSize * ubBlockSize;
    } else {
        tilingData_.singleBufferSize = tilingData_.singleBufferSize / config_.ubAlignSize * config_.ubAlignSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::CalcTilingKeyAndWorkspace()
{
    constexpr uint64_t DEFAULT_TILING_KEY = 100;
    workspaceSize_ = 0;
    tilingKey_ = DEFAULT_TILING_KEY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::WriteBackToContext()
{
    // 写入workspace size
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    // 写入启动核数
    context_->SetBlockDim(config_.kernelMode == RandomKernelMode::SIMD ? tilingData_.usedCoreNum :
                                                                         simtTilingData_.usedCoreNum);

    // 设置多核启动关系
    context_->SetScheduleMode(config_.isNeedSyncAll);

    // 设置tilingKey
    context_->SetTilingKey(tilingKey_);
    if (config_.DcacheSize != 0) {
        auto res = context_->SetLocalMemorySize(ubSize_);
        OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                    OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", static_cast<int64_t>(ubSize_)),
                    return ge::GRAPH_FAILED);
    }

    // 填充tilingData
    if (config_.kernelMode == RandomKernelMode::SIMD) {
        auto* tilingData = context_->GetTilingData<RandomUnifiedTilingDataStruct>();
        OP_CHECK_IF(tilingData == nullptr, OP_LOGE(opName_, "tilingData ptr is null"), return ge::GRAPH_FAILED);
        *tilingData = tilingData_;
    } else {
        auto* tilingData = context_->GetTilingData<RandomUnifiedSimtTilingDataStruct>();
        OP_CHECK_IF(tilingData == nullptr, OP_LOGE(opName_, "simtTilingData ptr is null"), return ge::GRAPH_FAILED);
        *tilingData = simtTilingData_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomTilingArch35::CheckTensor(const gert::CompileTimeTensorDesc* tensorDesc,
                                                const gert::Shape& tensorShape, const TensorCheckRule& rule,
                                                const std::string& tensorName)
{
    // 校验dtype
    if (!rule.dtypeSet.empty() && rule.dtypeSet.count(tensorDesc->GetDataType()) == 0) {
        std::string valueStr = Ops::Base::ToString(tensorDesc->GetDataType());
        std::string reasonMsg = "dtype not in allowed set";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), tensorName.c_str(), valueStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验shapeSize
    auto shapeSize = tensorShape.GetShapeSize();
    if (rule.shapeSize != -1 && shapeSize != rule.shapeSize) {
        std::string valueStr = std::to_string(shapeSize);
        std::string reasonMsg = "shape size must be " + std::to_string(rule.shapeSize);
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), tensorName.c_str(), valueStr.c_str(),
                                                  reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验维度数
    auto dimNum = tensorShape.GetDimNum();
    if (!rule.dimNumSet.empty() && rule.dimNumSet.count(dimNum) == 0) {
        std::string valueStr = std::to_string(dimNum);
        std::string reasonMsg = "dim num not in allowed set";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), tensorName.c_str(), valueStr.c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 自定义校验
    if (rule.customCheck && !rule.customCheck(context_)) {
        std::string reasonMsg = "custom check failed";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), tensorName.c_str(), "failed", reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
