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
 * \file coalesce_sparse_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "coalesce_sparse_tiling.h"

namespace {
constexpr uint64_t INT64_ID = 0;
constexpr uint64_t INT32_ID = 1;
constexpr uint64_t FP32_ID = 0;
constexpr uint64_t FP16_ID = 2;
constexpr uint64_t KEY_MODE1 = 6;
constexpr uint64_t KEY_MODE2 = 3;
constexpr uint64_t ASCEND950_SIMT_TILING_OFFSET = 100;
constexpr uint64_t AILGN256 = 256;
constexpr uint64_t AILGN32 = 32;
constexpr uint64_t MAXRPTIME = 4096;

} // namespace

namespace optiling {

class CoalesceSparseTiling {
public:
    explicit CoalesceSparseTiling(gert::TilingContext* context) : TilingContext(context) {};
    void TilingKeyCalcu(ge::DataType uniqueIndicesDtype, ge::DataType indicesDtype, ge::DataType valuesDtype);
    ge::graphStatus TilingDataCalcu(ge::DataType uniqueIndicesDtype, ge::DataType indicesDtype,
                                    ge::DataType valuesDtype, const gert::Shape& indicesShape,
                                    const gert::Shape& valueShape);
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    CoalesceSparseTilingData TilingData;
    gert::TilingContext* TilingContext = nullptr;
    uint64_t tiling_key = 0;
    uint64_t n = 0;
    uint64_t m = 0;
    uint64_t valueSize = 0;
    uint64_t taskNum = 0;
    uint64_t usedCoreNum = 0;
    uint64_t taskTail = 0;
    uint64_t moveOneSize = 0;
    uint64_t taskRepeatTimes = 0;
    uint64_t taskRepeatTail = 0;
    uint64_t taskTailRepeatTimes = 0;
    uint64_t taskTailRepeatTail = 0;
    uint64_t moveValueTimes = 0;
    uint64_t moveValueLen = 0;
    uint64_t moveValueTail = 0;
    uint64_t deterministicFlag = 0;
};

void CoalesceSparseTiling::TilingKeyCalcu(ge::DataType uniqueIndicesDtype, ge::DataType indicesDtype,
                                          ge::DataType valuesDtype)
{
    OP_LOGD(TilingContext, "GetTensorType.");
    OP_LOGD(TilingContext, "SetTilingKey.");
    auto platformInfo = TilingContext->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    bool isAscend950 = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    if (uniqueIndicesDtype == ge::DT_INT64) {
        tiling_key = INT64_ID * KEY_MODE1;
    } else {
        tiling_key = INT32_ID * KEY_MODE1;
    }
    if (indicesDtype == ge::DT_INT64) {
        tiling_key += INT64_ID * KEY_MODE2;
    } else {
        tiling_key += INT32_ID * KEY_MODE2;
    }
    if (valuesDtype == ge::DT_FLOAT) {
        tiling_key += FP32_ID;
    } else if (valuesDtype == ge::DT_INT32) {
        tiling_key += INT32_ID;
    } else {
        tiling_key += FP16_ID;
    }
    if (isAscend950) {
        tiling_key += ASCEND950_SIMT_TILING_OFFSET;
    }
}

ge::graphStatus CoalesceSparseTiling::TilingDataCalcu(ge::DataType uniqueIndicesDtype, ge::DataType indicesDtype,
                                                      ge::DataType valuesDtype, const gert::Shape& indicesShape,
                                                      const gert::Shape& valueShape)
{
    auto platformInfo = TilingContext->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t uniqueIndiceTypeSize = GetSizeByDataType(uniqueIndicesDtype);
    uint64_t indiceTypeSize = GetSizeByDataType(indicesDtype);
    uint64_t valueTypeSize = GetSizeByDataType(valuesDtype);
    auto ret = TilingContext->GetDeterministic();
    n = indicesShape.GetDim(0);
    m = indicesShape.GetDim(1);
    OP_LOGD(TilingContext, "Calculate valueSize.");
    valueSize = 1;
    uint64_t valueShapeSize = valueShape.GetDimNum();
    for (uint64_t i = 1; i < valueShapeSize; i++) {
        valueSize *= valueShape.GetDim(i);
    }
    OP_LOGD(TilingContext, "Init coreNum.");
    uint32_t coreNum = platformInfo->GetCoreNum();
    OP_LOGD(TilingContext, "Calculate taskNum.");
    OP_CHECK_IF(coreNum == 0, OP_LOGD(TilingContext, "Variale CoreNum should not be 0."), return ge::GRAPH_FAILED);
    taskNum = ((n - 1) / coreNum / AILGN256 + 1) * AILGN256;
    usedCoreNum = (n - 1) / taskNum + 1;
    taskTail = n % taskNum;
    if (taskTail == 0) {
        taskTail = taskNum;
    }
    bool isAscend950 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;
    if (isAscend950) {
        taskNum = (n + coreNum - 1) / coreNum;
        usedCoreNum = (n + taskNum - 1) / taskNum;
        taskTail = n % taskNum;
        if (taskTail == 0) {
            taskTail = taskNum;
        }
    }
    if (ret != 0) {
        usedCoreNum = 1;
        taskTail = n;
        deterministicFlag = 1;
    }
    moveValueLen = valueSize;
    if (moveValueLen > MAXRPTIME) {
        moveValueLen = MAXRPTIME;
    }
    OP_LOGD(TilingContext, "Init ubSize.");
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t indexUb = ubSize - (moveValueLen * valueTypeSize / AILGN32 + 1) * AILGN32;

    OP_CHECK_IF(indiceTypeSize == 0, OP_LOGD(TilingContext, "Variale indiceTypeSize should not be 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indiceTypeSize > AILGN32, OP_LOGD(TilingContext, "indiceTypeSize should not be greater than 32."),
                return ge::GRAPH_FAILED);
    uint64_t indicesSizeAlign32 = ((m - 1) / (AILGN32 / indiceTypeSize) + 1) * AILGN32;
    moveOneSize = (indexUb / (indicesSizeAlign32 + uniqueIndiceTypeSize)) / AILGN32 * AILGN32;
    if (moveOneSize >= MAXRPTIME) {
        moveOneSize = MAXRPTIME - AILGN32;
    }
    taskRepeatTimes = taskNum / moveOneSize;
    taskRepeatTail = taskNum % moveOneSize;
    taskTailRepeatTimes = taskTail / moveOneSize;
    taskTailRepeatTail = taskTail % moveOneSize;

    moveValueTimes = valueSize / moveValueLen;
    moveValueTail = valueSize % moveValueLen;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CoalesceSparseTiling::Init()
{
    OP_LOGD(TilingContext, "Tiling initing.");
    auto uniqueIndicesDtype = TilingContext->GetInputDesc(1)->GetDataType();
    auto indicesDtype = TilingContext->GetInputDesc(2)->GetDataType();
    auto valuesDtype = TilingContext->GetInputDesc(3)->GetDataType();
    auto indicesShape = TilingContext->GetInputShape(2)->GetStorageShape();
    auto valueShape = TilingContext->GetInputShape(3)->GetStorageShape();
    TilingKeyCalcu(uniqueIndicesDtype, indicesDtype, valuesDtype);
    auto ret = TilingDataCalcu(uniqueIndicesDtype, indicesDtype, valuesDtype, indicesShape, valueShape);
    OP_LOGD(TilingContext, "Tiling inited.");
    return ret;
}

ge::graphStatus CoalesceSparseTiling::RunKernelTiling()
{
    OP_LOGD(TilingContext, "Tiling start.");
    TilingContext->SetBlockDim(usedCoreNum);
    TilingContext->SetTilingKey(tiling_key);
    TilingData.set_usedCoreNum(usedCoreNum);
    TilingData.set_m(m);
    TilingData.set_valueSize(valueSize);
    TilingData.set_taskNum(taskNum);
    TilingData.set_taskTail(taskTail);
    TilingData.set_moveOneSize(moveOneSize);
    TilingData.set_taskRepeatTimes(taskRepeatTimes);
    TilingData.set_taskRepeatTail(taskRepeatTail);
    TilingData.set_taskTailRepeatTimes(taskTailRepeatTimes);
    TilingData.set_taskTailRepeatTail(taskTailRepeatTail);
    TilingData.set_moveValueTimes(moveValueTimes);
    TilingData.set_moveValueLen(moveValueLen);
    TilingData.set_moveValueTail(moveValueTail);
    TilingData.set_deterministicFlag(deterministicFlag);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = TilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(),
                            TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(TilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void CoalesceSparseTiling::TilingDataPrint()
{
    OP_LOGD(TilingContext, "usedCoreNum:%ld.", usedCoreNum);
    OP_LOGD(TilingContext, "m:%ld.", m);
    OP_LOGD(TilingContext, "valueSize:%ld.", valueSize);
    OP_LOGD(TilingContext, "taskNum:%ld.", taskNum);
    OP_LOGD(TilingContext, "taskTail:%ld.", taskTail);
    OP_LOGD(TilingContext, "moveOneSize:%ld.", moveOneSize);
    OP_LOGD(TilingContext, "taskRepeatTimes:%ld.", taskRepeatTimes);
    OP_LOGD(TilingContext, "taskRepeatTail:%ld.", taskRepeatTail);
    OP_LOGD(TilingContext, "taskTailRepeatTimes:%ld.", taskTailRepeatTimes);
    OP_LOGD(TilingContext, "taskTailRepeatTail:%ld.", taskTailRepeatTail);
    OP_LOGD(TilingContext, "moveValueTimes:%ld.", moveValueTimes);
    OP_LOGD(TilingContext, "moveValueLen:%ld.", moveValueLen);
    OP_LOGD(TilingContext, "moveValueTail:%ld.", moveValueTail);
    OP_LOGD(TilingContext, "deterministicFlag:%ld.", deterministicFlag);
}

static ge::graphStatus TilingCoalesceSparse(gert::TilingContext* context)
{
    CoalesceSparseTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}

struct CoalesceSparseCompileInfo {};
IMPL_OP_OPTILING(CoalesceSparse).Tiling(TilingCoalesceSparse);
} // namespace optiling
