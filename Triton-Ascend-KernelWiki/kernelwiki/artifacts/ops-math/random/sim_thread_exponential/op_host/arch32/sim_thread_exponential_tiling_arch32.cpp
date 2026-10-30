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
 * \file sim_thread_exponential_tiling_arch32.cpp
 * \brief
 */

#include "sim_thread_exponential_tiling_arch32.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_ascendc.h"

namespace {
constexpr uint32_t BLOCKSIZE = 256;
constexpr uint32_t BATCHNUMPERHANDLE = 6;
constexpr uint32_t UNROLLFACTOR = 4;
constexpr uint32_t INPUT_SELF_IDX = 0;
constexpr uint32_t OUTPUT_SELF_IDX = 0;
constexpr uint32_t FP32_TYPESIZE = 4;
constexpr uint64_t ATTR_0 = 0;
constexpr uint64_t ATTR_1 = 1;
constexpr uint64_t ATTR_2 = 2;
constexpr uint64_t ATTR_3 = 3;
constexpr int64_t THREAD_PER_PROCESSOR = 2048;
constexpr int64_t STREAM_PROCESSOR_COUNT = 78;
constexpr float SIM_THREAD_EXPONENTIAL_LOW = 0.0;
constexpr float SIM_THREAD_EXPONENTIAL_HIGH = 1.0;
constexpr uint32_t TILING_K_TWO = 2;
constexpr uint32_t TILING_K_THREE = 3;
constexpr uint32_t SHIFT_LEFT_32 = 32;
inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}
} // namespace

namespace optiling {
using namespace std;
void SimThreadExponentialTiling::PrintInfo()
{
    auto nodeName = context;
    OP_LOGD(nodeName, "----------------Start to print SimThreadExponentialTiling tiling data.----------------");
    OP_LOGD(nodeName, "key0 = %u.", tiling.get_key0());
    OP_LOGD(nodeName, "key1 = %u.", tiling.get_key1());
    OP_LOGD(nodeName, "offset_t_low = %u.", tiling.get_offset_t_low());
    OP_LOGD(nodeName, "offset_t_high = %u.", tiling.get_offset_t_high());
    OP_LOGD(nodeName, "useCoreNum = %u.", tiling.get_useCoreNum());
    OP_LOGD(nodeName, "batchNumPerCore = %u.", tiling.get_batchNumPerCore());
    OP_LOGD(nodeName, "batchNumTailCore = %u.", tiling.get_batchNumTailCore());
    OP_LOGD(nodeName, "batchNumTotal = %u.", tiling.get_batchNumTotal());
    OP_LOGD(nodeName, "numel = %ld.", tiling.get_numel());
    OP_LOGD(nodeName, "stepBlock = %u.", tiling.get_stepBlock());
    OP_LOGD(nodeName, "roundedSizeBlock = %u.", tiling.get_roundedSizeBlock());
    OP_LOGD(nodeName, "range = %f.", tiling.get_range());
    OP_LOGD(nodeName, "handleNumLoop = %u.", tiling.get_handleNumLoop());
    OP_LOGD(nodeName, "handleNumTail = %u.", tiling.get_handleNumTail());
    OP_LOGD(nodeName, "state = %u.", tiling.get_state());
    OP_LOGD(nodeName, "start = %f.", tiling.get_start());
    OP_LOGD(nodeName, "end = %f.", tiling.get_end());
    OP_LOGD(nodeName, "lambda = %f.", tiling.get_lambda());
    OP_LOGD(nodeName, "----------------print SimThreadExponentialTiling tiling data end.----------------");
}

void SimThreadExponentialTiling::SetTilingData()
{
    context->SetBlockDim(useCoreNum);
    tiling.set_key0(key0);
    tiling.set_key1(key1);
    tiling.set_offset_t_low(offset_t_low);
    tiling.set_offset_t_high(offset_t_high);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_batchNumPerCore(batchNumPerCore);
    tiling.set_batchNumTailCore(batchNumTailCore);
    tiling.set_batchNumTotal(batchNumTotal);
    tiling.set_numel(numel);
    tiling.set_stepBlock(stepBlock);
    tiling.set_roundedSizeBlock(roundedSizeBlock);
    tiling.set_range(range);
    tiling.set_handleNumLoop(handleNumLoop);
    tiling.set_handleNumTail(handleNumTail);
    tiling.set_state(state);
    tiling.set_start(start);
    tiling.set_end(end);
    tiling.set_lambda(lambda);
}

ge::graphStatus SimThreadExponentialTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    totalCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = static_cast<int64_t>(ubSizePlatForm);

    return ge::GRAPH_SUCCESS;
}

bool SimThreadExponentialTiling::GetDataTypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dataSizeType = 1;
            break;
        case ge::DT_BF16:
            dataSizeType = TILING_K_TWO;
            break;
        case ge::DT_FLOAT:
            dataSizeType = TILING_K_THREE;
            break;
        default:
            return false;
    }

    return true;
}

ge::graphStatus SimThreadExponentialTiling::GetInputTensorInfo()
{
    auto nodeName = context->GetNodeName();

    OP_CHECK_IF(count < 0, OP_LOGE(nodeName, "Count %ld must be greater than or equal to 0.", count),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(threadPerProcessor <= 0,
                OP_LOGE(nodeName, "ThreadPerProcessor %d must be greater than 0.", threadPerProcessor),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(streamProcessorCount <= 0,
                OP_LOGE(nodeName, "StreamProcessorCount %d must be greater than 0.", streamProcessorCount),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(start > end, // 如何获取start、end？
                OP_LOGE(nodeName, "Start %f must be less than or equal to end %f.", start, end),
                return ge::GRAPH_FAILED);

    // 获取第一个输入gradOut的信息
    auto selfShapePtr = context->GetInputShape(INPUT_SELF_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfShapePtr);
    auto selfShape = selfShapePtr->GetStorageShape();

    for (size_t i = 0; i < selfShape.GetDimNum(); ++i) {
        usrWorkspaceSize *= selfShape[i];
    }
    OP_CHECK_IF(usrWorkspaceSize != count,
                OP_LOGE(nodeName, "Count %ld must be equal to the product of the elements in selfShape.", count),
                return ge::GRAPH_FAILED);
    usrWorkspaceSize = Ceil(usrWorkspaceSize, BATCHNUMPERHANDLE * BLOCKSIZE);
    usrWorkspaceSize *= FP32_TYPESIZE;

    auto selfDesc = context->GetInputDesc(INPUT_SELF_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfDesc);
    selfDType = selfDesc->GetDataType();
    GetDataTypeKey(selfDType);
    OP_CHECK_IF(GetDataTypeKey(selfDType) == false,
                OP_LOGE(nodeName, "The dtype of input self must be in [float32, float16, bfloat16]."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SimThreadExponentialTiling::Tiling4Block()
{
    auto nodeName = context->GetNodeName();

    uint32_t default_block_num = threadPerProcessor / BLOCKSIZE * streamProcessorCount;
    batchNumTotal = Min<uint32_t>((count + BLOCKSIZE - 1) / BLOCKSIZE, default_block_num);
    stepNum = BLOCKSIZE * batchNumTotal * UNROLLFACTOR;
    stepBlock = batchNumTotal * UNROLLFACTOR;
    roundedSizeNum = ((numel - 1) / stepNum + 1) * stepNum;
    roundedSizeBlock = roundedSizeNum / BLOCKSIZE;
    range = end - start;

    key0 = static_cast<uint32_t>(seed);
    key1 = static_cast<uint32_t>(seed >> SHIFT_LEFT_32);
    state = 0;
    state += offset & static_cast<uint64_t>(ATTR_3);
    uint64_t offset_t = offset / UNROLLFACTOR;
    if (state > ATTR_3) {
        offset_t += 1;
        state -= UNROLLFACTOR;
    }
    offset_t_low = static_cast<uint32_t>(offset_t);
    offset_t_high = static_cast<uint32_t>(offset_t >> SHIFT_LEFT_32);

    // 分核计算
    useCoreNum = static_cast<int64_t>(
        Ops::Base::CeilDiv(batchNumTotal, Ops::Base::CeilDiv(batchNumTotal, totalCoreNum)));
    OP_CHECK_IF(useCoreNum == 0, OP_LOGE(nodeName, "useCoreNum %u must be not equal to 0.", useCoreNum),
                return ge::GRAPH_FAILED);
    // useCoreNum = static_cast<int64_t>(CeilDiv(batchNumTotal, CeilDiv(batchNumTotal, totalCoreNum)));
    batchNumPerCore = (batchNumTotal + useCoreNum - 1) / useCoreNum;
    batchNumTailCore = batchNumTotal - (useCoreNum - 1) * batchNumPerCore;
    // 分块计算
    handleNumLoop = batchNumPerCore / BATCHNUMPERHANDLE;
    handleNumTail = batchNumPerCore - handleNumLoop * BATCHNUMPERHANDLE;

    OP_CHECK_IF(batchNumPerCore <= 0, OP_LOGE(nodeName, "batchNumPerCore %u must be greater than 0.", batchNumPerCore),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SimThreadExponentialTiling::SetAttrParams()
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* countPtr = attrs->GetAttrPointer<int64_t>(ATTR_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, countPtr);
    count = static_cast<int64_t>(*countPtr);
    start = SIM_THREAD_EXPONENTIAL_LOW;
    end = SIM_THREAD_EXPONENTIAL_HIGH; // sim_expinential功能与sim_random_uniform类似，start固定为0.0，end固定为1.0
    const float* lambdaPtr = attrs->GetAttrPointer<float>(ATTR_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, lambdaPtr);
    lambda = static_cast<float>(*lambdaPtr);
    OP_CHECK_IF(lambda == 0,
                OP_LOGE(context->GetNodeName(), "lambda is the denominator and cannot be zero, but get %f.", lambda),
                return ge::GRAPH_FAILED);
    const int64_t* seedPtr = attrs->GetAttrPointer<int64_t>(ATTR_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, seedPtr);
    seed = static_cast<uint64_t>(*seedPtr);
    const int64_t* offsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetPtr);
    offset = static_cast<uint64_t>(*offsetPtr);
    threadPerProcessor = THREAD_PER_PROCESSOR;
    streamProcessorCount = STREAM_PROCESSOR_COUNT;
    numel = count;

    return ge::GRAPH_SUCCESS;
}

void SimThreadExponentialTiling::SetTilingKey() { tilingKey_ = dataSizeType; }

uint64_t SimThreadExponentialTiling::GetTilingKey() { return tilingKey_; }

ge::graphStatus SimThreadExponentialTiling::DoTiling()
{
    auto nodeName = context->GetNodeName();

    OP_CHECK_IF(SetAttrParams() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "SetAttrParams failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetInputTensorInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Tiling4Block() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling4Block failed."),
                return ge::GRAPH_FAILED);
    SetTilingData();

    SetTilingKey();
    context->SetTilingKey(GetTilingKey());

    PrintInfo();
    // 固定写法
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace
        [0] = usrWorkspaceSize +
              ascendcPlatform
                  .GetLibApiWorkSpaceSize(); // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
                                             // 该workspace作为中转空间用来放T2类型数据

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SimThreadExponential(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling4SimThreadExponential running begin.");

    SimThreadExponentialTiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4SimThreadExponential(gert::TilingParseContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepare4SimThreadExponential running end.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SimThreadExponential)
    .Tiling(Tiling4SimThreadExponential)
    .TilingParse<Tiling4SimThreadExponentialCompileInfo>(TilingPrepare4SimThreadExponential);
} // namespace optiling
