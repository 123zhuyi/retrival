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
 * \file lin_space_tiling.cpp
 * \brief
 */

#include "lin_space_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
static const size_t INPUT_IDX_START = 0;
static const size_t INPUT_IDX_STOP = 1;
static const size_t INPUT_IDX_NUM = 2;
static const size_t POWER_BASE_NUM = 2;
static const int32_t INT16_BITS_NUM = 16;
static const int32_t BLOCK_SIZE = 32;
static const int64_t MATRIX_SIZE = 256;
static const int32_t OUT_SIZE = 16 * 1024;
static const int32_t WORK_SPACE_SIZE = 32;

class LinSpaceMemBaseTilingClass : public Ops::Base::TilingBaseClass
{
public:
    explicit LinSpaceMemBaseTilingClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        Ops::Base::TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        auto platformInfo = context_->GetPlatformInfo();
        if (platformInfo != nullptr) {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            npuArch_ = ascendcPlatform.GetCurNpuArch();
        } else {
            auto compileInfoPtr = reinterpret_cast<const LinSpaceCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
            npuArch_ = compileInfoPtr->npuArch;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return (!Ops::Base::IsRegbaseSocVersion(context_));
    }

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }

    NpuArch npuArch_;
};

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

inline static int64_t GetBaseNum(int64_t value)
{
    int64_t valueNum = 0;
    if (value == 0) {
        return valueNum;
    }
    for (int64_t idex = 1; idex <= value; idex *= POWER_BASE_NUM) {
        valueNum++;
    }
    return valueNum;
}

inline static ge::graphStatus LinSpaceSetTilingData(gert::TilingContext* context, LinSpaceTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

inline static int64_t CalcAlignNumPerCore(const ge::DataType outputDtype, const gert::TilingContext* context)
{
    int32_t typeSize = ge::GetSizeByDataType(outputDtype);
    OP_CHECK_IF(
        (typeSize <= 0),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "output", Ops::Base::ToString(outputDtype).c_str(),
            "The dtype size of output must be greater than 0"),
        return -1);
    return BLOCK_SIZE / typeSize;
}

inline static int64_t CalcMaxOutNum(const ge::DataType outDataType, const gert::TilingContext* context)
{
    ge::DataType calcDataType = outDataType;
    
    if (outDataType == ge::DT_BF16 || outDataType == ge::DT_INT32) {
        calcDataType = ge::DT_FLOAT;
    } else if (outDataType == ge::DT_INT8 || outDataType == ge::DT_UINT8 || 
               outDataType == ge::DT_INT16) {
        calcDataType = ge::DT_FLOAT16;
    }
    
    int32_t outTypeSize = ge::GetSizeByDataType(calcDataType);
    OP_CHECK_IF(
        (outTypeSize <= 0),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "output", Ops::Base::ToString(calcDataType).c_str(),
            "The dtype size of output must be greater than 0"),
        return -1);
    return OUT_SIZE / outTypeSize;
}

static bool InputTypeIsInvalid(const gert::TilingContext* context)
{
    auto startDes = context->GetInputDesc(INPUT_IDX_START);
    OP_CHECK_NULL_WITH_CONTEXT(context, startDes);
    auto dStart = startDes->GetDataType();
    auto stopDsc = context->GetInputDesc(INPUT_IDX_STOP);
    OP_CHECK_NULL_WITH_CONTEXT(context, stopDsc);
    auto dStop = stopDsc->GetDataType();
    auto numDsc = context->GetInputDesc(INPUT_IDX_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, numDsc);
    auto dNum = numDsc->GetDataType();

    bool dStartIsInvalid = (dStart != ge::DT_INT32) && (dStart != ge::DT_FLOAT) && (dStart != ge::DT_FLOAT16) &&
                           (dStart != ge::DT_BF16) && (dStart != ge::DT_INT8) && (dStart != ge::DT_UINT8) &&
                           (dStart != ge::DT_INT16);
    OP_CHECK_IF(
        dStartIsInvalid,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "start", Ops::Base::ToString(dStart).c_str(),
            "float32, float16, bfloat16, int32, int16, int8 or uint8"),
        return true);

    bool dStopIsInvalid = (dStop != ge::DT_INT32) && (dStop != ge::DT_FLOAT) && (dStop != ge::DT_FLOAT16) &&
                          (dStop != ge::DT_BF16) && (dStop != ge::DT_INT8) && (dStop != ge::DT_UINT8) &&
                          (dStop != ge::DT_INT16);
    OP_CHECK_IF(
        dStopIsInvalid,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "stop", Ops::Base::ToString(dStop).c_str(),
            "float32, float16, bfloat16, int32, int16, int8 or uint8"),
        return true);
    bool dNumIsInvalid = (dNum != ge::DT_INT32) && (dNum != ge::DT_INT64);
    OP_CHECK_IF(
        dNumIsInvalid,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "num", Ops::Base::ToString(dNum).c_str(), "int32 or int64"),
        return true);
    // any input dtype is neither int32 nor float, invalid
    return dStartIsInvalid || dStopIsInvalid || dNumIsInvalid;
}

template <typename T>
static ge::graphStatus LinSpaceGetConstValue(
    gert::TilingContext* context, const gert::Tensor* tensor, T& value, ge::DataType dataType)
{
    if (dataType == ge::DT_INT64) {
        const int64_t* const_data_ptr = tensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        float f32 = static_cast<float>(*const_data_ptr);
        value = static_cast<T>(f32);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%ld", *const_data_ptr);
    } else if (dataType == ge::DT_INT32) {
        const int32_t* const_data_ptr = tensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        value = static_cast<T>(*const_data_ptr);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
    } else if (dataType == ge::DT_FLOAT) {
        const float* const_data_ptr = tensor->GetData<float>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        value = static_cast<T>(*const_data_ptr);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%f", *const_data_ptr);
    } else if (dataType == ge::DT_FLOAT16) {
        const Ops::Base::fp16_t* const_data_ptr = tensor->GetData<Ops::Base::fp16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        float f32 = static_cast<float>(*const_data_ptr);
        value = static_cast<T>(f32);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%f", static_cast<float>(*const_data_ptr));
    } else if (dataType == ge::DT_INT16) {
        const int16_t* const_data_ptr = tensor->GetData<int16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        value = static_cast<T>(*const_data_ptr);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
    } else if (dataType == ge::DT_BF16) {
        const int16_t* const_data_ptr = tensor->GetData<int16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        int32_t f32hex = (static_cast<int32_t>(*const_data_ptr)) << INT16_BITS_NUM;
        value = static_cast<T>((reinterpret_cast<float&>(f32hex)));
    } else if (dataType == ge::DT_INT8) {
        const int8_t* const_data_ptr = tensor->GetData<int8_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        value = static_cast<T>(*const_data_ptr);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
    } else if (dataType == ge::DT_UINT8) {
        const uint8_t* const_data_ptr = tensor->GetData<uint8_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
        value = static_cast<T>(*const_data_ptr);
        OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
    } else {
        // do nothing, impossible
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConstNum(
    gert::TilingContext* context, const gert::Tensor* tensor_num, int64_t& num, int64_t input_index)
{
    auto numDesc = context->GetInputDesc(input_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, numDesc);
    const ge::DataType dataType = numDesc->GetDataType();
    switch (dataType) {
        case ge::DT_INT32: {
            int32_t num_32(0);
            OP_CHECK_IF(
                LinSpaceGetConstValue<int32_t>(context, tensor_num, num_32, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get num const value fail."), return ge::GRAPH_FAILED);
            num = (int64_t)num_32;
            break;
        }
        case ge::DT_INT64:
            OP_CHECK_IF(
                LinSpaceGetConstValue<int64_t>(context, tensor_num, num, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get const value fail."), return ge::GRAPH_FAILED);
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "num",
                Ops::Base::ToString(dataType).c_str(), "int32 or int64");
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConstStartOrStop(
    gert::TilingContext* context, const gert::Tensor* tensor_index, float& index, int64_t input_index)
{
    auto inputDesc = context->GetInputDesc(input_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    switch (dataType) {
        case ge::DT_INT32: {
            int32_t index_32(0);
            LinSpaceGetConstValue<int32_t>(context, tensor_index, index_32, dataType);
            index = static_cast<float>(index_32);
            break;
        }
        case ge::DT_FLOAT:
        case ge::DT_BF16: {
            LinSpaceGetConstValue<float>(context, tensor_index, index, dataType);
            break;
        }
        case ge::DT_FLOAT16: {
            Ops::Base::fp16_t index_16(0);
            LinSpaceGetConstValue<Ops::Base::fp16_t>(context, tensor_index, index_16, dataType);
            index = static_cast<float>(index_16);
            break;
        }
        case ge::DT_INT16: {
            int16_t index_16(0);
            LinSpaceGetConstValue<int16_t>(context, tensor_index, index_16, dataType);
            index = static_cast<float>(index_16);
            break;
        }
        case ge::DT_INT8: {
            int8_t index_8(0);
            LinSpaceGetConstValue<int8_t>(context, tensor_index, index_8, dataType);
            index = static_cast<float>(index_8);
            break;
        }
        case ge::DT_UINT8: {
            uint8_t index_8(0);
            LinSpaceGetConstValue<uint8_t>(context, tensor_index, index_8, dataType);
            index = static_cast<float>(index_8);
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(),
                (input_index == static_cast<int64_t>(INPUT_IDX_START)) ? "start" : "stop",
                Ops::Base::ToString(dataType).c_str(),
                "float32, float16, bfloat16, int32, int16, int8 or uint8");
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetLoopNumForLinSpace(
    const gert::TilingContext* context, LinSpaceTilingData& tilingData, const ge::DataType outDataType)
{
    int64_t maxOutNum = CalcMaxOutNum(outDataType, context);
    int64_t matrixLen = tilingData.get_numPerCore() <= MATRIX_SIZE ? tilingData.get_numPerCore() : MATRIX_SIZE;
    tilingData.set_matrixLen(matrixLen);
    tilingData.set_outerLoopNum(CeilDiv(tilingData.get_numPerCore(), maxOutNum));
    tilingData.set_outerLoopNumTail(tilingData.get_numPerCore() % maxOutNum);
    tilingData.set_outerTailLoopNum(CeilDiv(tilingData.get_tailNum(), maxOutNum));
    tilingData.set_outerTailLoopNumTail(tilingData.get_tailNum() % maxOutNum);

    if (tilingData.get_numPerCore() <= MATRIX_SIZE) {
        tilingData.set_innerLoopNum(0);
        tilingData.set_innerLoopTail(0);
        tilingData.set_innerTailLoopNum(0);
        tilingData.set_innerTailLoopTail(0);
    } else {
        tilingData.set_innerLoopNum(tilingData.get_numPerCore() / MATRIX_SIZE / POWER_BASE_NUM);
        tilingData.set_innerLoopTail(
            tilingData.get_numPerCore() - (MATRIX_SIZE << GetBaseNum(tilingData.get_innerLoopNum())));
        tilingData.set_innerTailLoopNum(tilingData.get_tailNum() / MATRIX_SIZE / POWER_BASE_NUM);
        tilingData.set_innerTailLoopTail(
            tilingData.get_tailNum() > MATRIX_SIZE ?
                (tilingData.get_tailNum() - (MATRIX_SIZE << GetBaseNum(tilingData.get_innerTailLoopNum()))) :
                0);
    }
    OP_LOGD(
        context->GetNodeName(),
        "tilingData is matrixLen:%ld, outerLoopNum:%ld, outerLoopNumTail:%ld, \
          outerTailLoopNum:%ld, outerTailLoopNumTail:%ld, innerLoopNum:%ld, innerLoopTail:%ld, \
          innerTailLoopNum:%ld, innerTailLoopTail:%ld",
        tilingData.get_matrixLen(), tilingData.get_outerLoopNum(), tilingData.get_outerLoopNumTail(),
        tilingData.get_outerTailLoopNum(), tilingData.get_outerTailLoopNumTail(), tilingData.get_innerLoopNum(),
        tilingData.get_innerLoopTail(), tilingData.get_innerTailLoopNum(), tilingData.get_innerTailLoopTail());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingTilingKeyOneCore(
    const gert::TilingContext* context, LinSpaceTilingData& tilingData, const ge::DataType outDataType)
{
    switch (outDataType) {
        case ge::DT_FLOAT:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_103));
            break;
        case ge::DT_FLOAT16:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_203));
            break;
        case ge::DT_INT8:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_303));
            break;
        case ge::DT_UINT8:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_403));
            break;
        case ge::DT_INT16:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_503));
            break;
        case ge::DT_INT32:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_603));
            break;
        case ge::DT_BF16:
            tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_703));
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "output",
                Ops::Base::ToString(outDataType).c_str(),
                "float32, float16, bfloat16, int32, int16, int8 or uint8");
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingTilingKeyForLinSpace(
    gert::TilingContext* context, LinSpaceTilingData& tilingData, const ge::DataType outDataType)
{
    int64_t maxOutNum = CalcMaxOutNum(outDataType, context);
    if (tilingData.get_realCoreNum() == 1) {
        OP_CHECK_IF(
            SetTilingTilingKeyOneCore(context, tilingData, outDataType) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "set tilingKey fail."), return ge::GRAPH_FAILED);
    } else {
        switch (outDataType) {
            case ge::DT_FLOAT:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_101) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_102));
                break;
            case ge::DT_FLOAT16:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_201) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_202));
                break;
            case ge::DT_INT8:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_301) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_302));
                break;
            case ge::DT_UINT8:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_401) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_402));
                break;
            case ge::DT_INT16:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_501) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_502));
                break;
            case ge::DT_INT32:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_601) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_602));
                break;
            case ge::DT_BF16:
                tilingData.set_tilingKey(
                    tilingData.get_numPerCore() <= maxOutNum ? static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_701) :
                                                               static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_702));
                break;
            default:
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "output",
                    Ops::Base::ToString(outDataType).c_str(),
                    "float32, float16, bfloat16, int32, int16, int8 or uint8");
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingDataForLinSpace(
    gert::TilingContext* context, LinSpaceTilingData& tilingData, const ge::DataType outDataType)
{
    OP_CHECK_IF(
        SetTilingTilingKeyForLinSpace(context, tilingData, outDataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "set loopNum fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SetLoopNumForLinSpace(context, tilingData, outDataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "set loopNum fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        LinSpaceSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "LinSpaceSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;

    OP_LOGD(
        context->GetNodeName(), "tilingData is tilingKey:%ld, realCoreNum:%ld, numPerCore:%ld, tailNum:%ld",
        tilingData.get_tilingKey(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
        tilingData.get_tailNum());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingBasicInfo(
    const gert::TilingContext* context, LinSpaceTilingData& tilingData, const float& start, const float& stop,
    const int64_t& num)
{
    tilingData.set_start(start);
    tilingData.set_stop(stop);
    tilingData.set_num(num);
    if (num > 1) {
        tilingData.set_scalar((stop - start) / (num - 1));
    } else {
        tilingData.set_scalar(0);
    }
    OP_LOGD(
        context->GetNodeName(), "tilingData is start:%f, stop:%f, num:%ld, scalar:%f", tilingData.get_start(),
        tilingData.get_stop(), tilingData.get_num(), tilingData.get_scalar());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LinSpace(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4LinSpace enter.");

    auto compileInfo = context->GetCompiledInfo<LinSpaceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4LinSpace fail to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4LinSpaceFlat fail to get ub size."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LinSpaceMemBaseTilingClass::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling4LinSpace enter.");
    auto tensor_start = context_->GetInputTensor(INPUT_IDX_START);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor_start);
    auto tensor_stop = context_->GetInputTensor(INPUT_IDX_STOP);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor_stop);
    auto tensor_num = context_->GetInputTensor(INPUT_IDX_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor_num);
    OP_CHECK_IF(
        InputTypeIsInvalid(context_), OP_LOGE(context_->GetNodeName(), "input dtype is invalid."),
        return ge::GRAPH_FAILED);

    LinSpaceTilingData tilingData;
    float start(0);
    float stop(0);
    int64_t num(0);
    OP_CHECK_IF(
        GetConstNum(context_, tensor_num, num, INPUT_IDX_NUM) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set tilingData num fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetConstStartOrStop(context_, tensor_start, start, INPUT_IDX_START) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set tilingData start fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetConstStartOrStop(context_, tensor_stop, stop, INPUT_IDX_STOP) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set tilingData stop fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SetTilingBasicInfo(context_, tilingData, start, stop, num) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set basic info fail."), return ge::GRAPH_FAILED);

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();

    int64_t alignNumPerCore = CalcAlignNumPerCore(outputDtype, context_);
    auto compileInfo = reinterpret_cast<const LinSpaceCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    uint32_t coreNum = compileInfo->totalCoreNum;
    int64_t tmpRealCoreNum = tilingData.get_num() < coreNum ? tilingData.get_num() : coreNum;
    int64_t tmpNumPerCore = CeilDiv(tilingData.get_num(), tmpRealCoreNum);

    tilingData.set_numPerCore(CeilDiv(tmpNumPerCore, alignNumPerCore) * alignNumPerCore);
    tilingData.set_realCoreNum(CeilDiv(tilingData.get_num(), tilingData.get_numPerCore()));
    int64_t tailNum = tilingData.get_num() - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore();
    tilingData.set_tailNum(tailNum);
    OP_CHECK_IF(
        SetTilingDataForLinSpace(context_, tilingData, outputDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set TilingKey fail."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Tiling4LinSpace exit.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LinSpace(gert::TilingContext* context)
{
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

REGISTER_OPS_TILING_TEMPLATE(LinSpace, LinSpaceMemBaseTilingClass, 1000);

IMPL_OP_OPTILING(LinSpace)
    .Tiling(Tiling4LinSpace)
    .TilingInputsDataDependency({0, 1, 2})
    .TilingParse<LinSpaceCompileInfo>(TilingPrepare4LinSpace);
} // namespace optiling