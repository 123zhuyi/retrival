/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_tiling_arch35.cpp
 * \brief
 */
#include <vector>
#include <cmath>
#include "range_tiling.h"
#include "range_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace optiling {

constexpr static int64_t CORE_MINEST_NUM = 128;
constexpr static int64_t RESERVED_UB_SIZE = 1024;
constexpr static int64_t DOUBLE_UB_SIZE = 2;
constexpr static int32_t UB_CALCU_TWO_RADIO = 2;
constexpr static int32_t UB_CALCU_THREE_RADIO = 3;
constexpr static int32_t UB_CALCU_FIVE_RADIO = 5;

constexpr static int32_t INDEX_INPUT_START = 0;
constexpr static int32_t INDEX_INPUT_END = 1;
constexpr static int32_t INDEX_INPUT_STEP = 2;
constexpr static int32_t INDEX_OUTPUT_OUT = 0;

static inline int64_t Align128CeilSize(int64_t value)
{
    return static_cast<int64_t>((value + CORE_MINEST_NUM - 1) / CORE_MINEST_NUM * CORE_MINEST_NUM);
}

static inline int64_t Align128FloorSize(int64_t value)
{
    return static_cast<int64_t>((value / CORE_MINEST_NUM) * CORE_MINEST_NUM);
}

static ge::graphStatus CheckDtypeIsInvalid(
    gert::TilingContext* context, ge::DataType start, ge::DataType limit, ge::DataType delta, ge::DataType output)
{
    std::set<ge::DataType> outputDtype = {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64};
    std::set<ge::DataType> inputDtype = {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16,
                                         ge::DT_BF16,  ge::DT_INT64, ge::DT_DOUBLE};

    OP_CHECK_IF(
        inputDtype.count(start) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "start", Ops::Base::ToString(start).c_str(),
            "int32, int64, float32, float16, bfloat16 or double"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputDtype.count(limit) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "limit", Ops::Base::ToString(limit).c_str(),
            "int32, int64, float32, float16, bfloat16 or double"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputDtype.count(delta) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "delta", Ops::Base::ToString(delta).c_str(),
            "int32, int64, float32, float16, bfloat16 or double"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputDtype.count(output) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", Ops::Base::ToString(output).c_str(),
            "int32, int64, float32, float16 or bfloat16"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcRangeTilingParam(
    const gert::TilingContext* context, RangeTilingParam& tilingParam, DataType dataType)
{
    OP_LOGD("[CalcRangeTilingParam]", "TilingRange Enter CalcRangeTilingParam funtion.");

    int64_t numOfPerCore = tilingParam.totalElementNum;
    int64_t usedCoreNum = 1;
    if (tilingParam.totalElementNum > CORE_MINEST_NUM) {
        numOfPerCore =
            Align128CeilSize((tilingParam.totalElementNum + tilingParam.totalCoreNum - 1) / tilingParam.totalCoreNum);
        OP_CHECK_IF(
            numOfPerCore == 0, OP_LOGE(context->GetNodeName(), "numOfPerCore should not be zero."),
            return ge::GRAPH_FAILED);
        usedCoreNum = min((tilingParam.totalElementNum + numOfPerCore - 1) / numOfPerCore, tilingParam.totalCoreNum);
    }
    int64_t numOfTailCore = tilingParam.totalElementNum - (usedCoreNum - 1) * numOfPerCore;

    int64_t ubCap = tilingParam.hardwareUbSize / DOUBLE_UB_SIZE;
    int64_t ubNum = (ubCap - RESERVED_UB_SIZE) / ge::GetSizeByDataType(dataType);

    int64_t ubOneLoopNum = Align128FloorSize(ubNum / UB_CALCU_TWO_RADIO);
    if (dataType == DT_FLOAT) {
        ubOneLoopNum = Align128FloorSize(ubNum / UB_CALCU_THREE_RADIO);
    } else if (dataType == DT_BF16 || dataType == DT_FLOAT16) {
        ubOneLoopNum = Align128FloorSize(ubNum / UB_CALCU_FIVE_RADIO);
    }

    int64_t perOfPerCore = (numOfPerCore < ubOneLoopNum) ? numOfPerCore : ubOneLoopNum;
    int64_t loopOfPerCore = Ops::Base::CeilDiv(numOfPerCore, perOfPerCore);
    int64_t tailOfPerCore = numOfPerCore - perOfPerCore * (loopOfPerCore - 1);
    int64_t perOfTailCore = (numOfTailCore < ubOneLoopNum) ? numOfTailCore : ubOneLoopNum;
    int64_t loopOfTailCore = Ops::Base::CeilDiv(numOfTailCore, perOfTailCore);
    int64_t tailOfTailCore = numOfTailCore - perOfTailCore * (loopOfTailCore - 1);

    tilingParam.numOfPerCore = numOfPerCore;
    tilingParam.loopOfPerCore = loopOfPerCore;
    tilingParam.perOfPerCore = perOfPerCore;
    tilingParam.tailOfPerCore = tailOfPerCore;
    tilingParam.numOfTailCore = numOfTailCore;
    tilingParam.loopOfTailCore = loopOfTailCore;
    tilingParam.perOfTailCore = perOfTailCore;
    tilingParam.tailOfTailCore = tailOfTailCore;
    tilingParam.ubOneLoopNum = ubOneLoopNum;
    tilingParam.usedCoreNum = usedCoreNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RangeRegBaseTilingClass::DoOpTiling()
{
    OP_LOGD(context_, "[Range] RangeTilingForAscendC running begin.");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_, "Context is nullptr"), return ge::GRAPH_FAILED);

    auto start = context_->GetInputTensor(INDEX_INPUT_START);
    OP_CHECK_NULL_WITH_CONTEXT(context_, start);
    auto limit = context_->GetInputTensor(INDEX_INPUT_END);
    OP_CHECK_NULL_WITH_CONTEXT(context_, limit);
    auto delta = context_->GetInputTensor(INDEX_INPUT_STEP);
    OP_CHECK_NULL_WITH_CONTEXT(context_, delta);
    auto tensorY = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorY);

    auto dtypeStart = start->GetDataType();
    auto dtypeLimit = limit->GetDataType();
    auto dtypeDelta = delta->GetDataType();
    auto dtypeY = tensorY->GetDataType();
    auto ret = CheckDtypeIsInvalid(context_, dtypeStart, dtypeLimit, dtypeDelta, dtypeY);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto compileInfo = reinterpret_cast<const RangeCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    tilingParam_.totalCoreNum = compileInfo->totalCoreNum;
    tilingParam_.hardwareUbSize = compileInfo->ubSize;

    uint64_t outSize = 0;
    switch (outDataType_) {
        case ge::DT_INT32:
        case ge::DT_INT64: {
            OP_CHECK_IF(
                CalculateOutputSize<int64_t>(context_, start, limit, delta, outSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalculateOutputSize fail."), return ge::GRAPH_FAILED);
            break;
        }
        case ge::DT_FLOAT: {
            OP_CHECK_IF(
                CalculateOutputSize<double>(context_, start, limit, delta, outSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalculateOutputSize fail."), return ge::GRAPH_FAILED);
            break;
        }
        default: {
            OP_CHECK_IF(
                CalculateOutputSize<float>(context_, start, limit, delta, outSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "append tiling args fail."), return ge::GRAPH_FAILED);
            break;
        }
    }
    // 计算元素总个数
    tilingParam_.totalElementNum = static_cast<int64_t>(outSize);

    // 设置Range算子中的参数
    OP_CHECK_IF(
        CalcRangeTilingParam(context_, tilingParam_, outDataType_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "SetRangeTilingParam fail"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling