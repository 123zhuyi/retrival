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
 * \file lin_space_tiling_arch35.cpp
 * \brief
 */
#include "lin_space_tiling_arch35.h"
#include "log/log.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
constexpr static int64_t CORE_MINEST_NUM = 128;
constexpr static int64_t RESERVED_UB_SIZE = 1024;
constexpr static int64_t DOUBLE_UB_SIZE = 2;
constexpr static int32_t UB_CALCU_BIT_THIRTYTWO_RADIO = 3;
constexpr static int32_t UB_CALCU_BIT_SIXTEEN_RADIO = 4;
constexpr static int32_t UB_CALCU_BIT_EIGHT_RADIO = 6;

constexpr static int32_t INDEX_INPUT_START = 0;
constexpr static int32_t INDEX_INPUT_STOP = 1;
constexpr static int32_t INDEX_INPUT_NUM = 2;
constexpr static int32_t INDEX_OUTPUT_OUT = 0;
constexpr static size_t WORKSPACE_COUNT = 1;
constexpr static const int32_t INT16_BITS_NUM = 16;

class LinSpaceRegbaseTilingClass : public Ops::Base::TilingBaseClass
{
public:
    explicit LinSpaceRegbaseTilingClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        Ops::Base::TilingBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    void SetStep();

    ge::graphStatus DoOpTiling() override;
    ge::DataType outDataType_;
    NpuArch npuArch_;

private:
    LinSpaceRegbaseTilingData tilingData_;
    LinSpaceRegbaseTilingParam tilingParam_;

    int64_t num;
};

static inline int64_t Align128CeilSize(int64_t value)
{
    return static_cast<int64_t>((value + CORE_MINEST_NUM - 1) / CORE_MINEST_NUM * CORE_MINEST_NUM);
}

static inline int64_t Align128FloorSize(int64_t value)
{
    return static_cast<int64_t>((value / CORE_MINEST_NUM) * CORE_MINEST_NUM);
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
                (input_index == static_cast<int64_t>(INDEX_INPUT_START)) ? "start" : "stop",
                Ops::Base::ToString(dataType).c_str(),
                "float32, float16, bfloat16, int32, int16, int8 or uint8");
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtypeIsValid(
    gert::TilingContext* context, ge::DataType start, ge::DataType stop, ge::DataType num, ge::DataType output)
{
    std::set<ge::DataType> numDtype = {ge::DT_INT32, ge::DT_INT64};
    std::set<ge::DataType> otherDtype = {ge::DT_UINT8, ge::DT_INT8,    ge::DT_INT16, ge::DT_INT32,
                                         ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

    OP_CHECK_IF(
        otherDtype.count(start) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "start", Ops::Base::ToString(start).c_str(),
            "uint8, int8, int16, int32, float32, float16 or bf16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        otherDtype.count(stop) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "stop", Ops::Base::ToString(stop).c_str(),
            "uint8, int8, int16, int32, float32, float16 or bf16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        otherDtype.count(output) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "output", Ops::Base::ToString(output).c_str(),
            "uint8, int8, int16, int32, float32, float16 or bf16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numDtype.count(num) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "num", Ops::Base::ToString(num).c_str(),
            "int32 or int64"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcLinSpaceTilingParam(LinSpaceRegbaseTilingParam& tilingParam, ge::DataType outDtype)
{
    OP_LOGD("[CalcLinSpaceTilingParam]", "TilingLinSpace Enter CalcLinSpaceTilingParam funtion.");

    // 分核
    int64_t numOfPerCore = tilingParam.num > 0 ? tilingParam.num : 0;
    int64_t usedCoreNum = 1;
    if (tilingParam.num > CORE_MINEST_NUM) {
        numOfPerCore = Align128CeilSize((tilingParam.num + tilingParam.totalCoreNum - 1) / tilingParam.totalCoreNum);
        usedCoreNum = std::min((tilingParam.num + numOfPerCore - 1) / numOfPerCore, tilingParam.totalCoreNum);
    }
    int64_t numOfTailCore = tilingParam.num > 0 ? (tilingParam.num - (usedCoreNum - 1) * numOfPerCore) : 0;

    // ub切分
    int32_t outputSize = std::max(INDEX_INPUT_STOP, ge::GetSizeByDataType(outDtype));
    int64_t ubNum = std::max(static_cast<int64_t>(INDEX_INPUT_STOP), (tilingParam.totalUbSize - RESERVED_UB_SIZE));
    if (outputSize <= 0 || ubNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    ubNum = ubNum / outputSize;
    int64_t ubOneLoopNum = Align128FloorSize(ubNum / UB_CALCU_BIT_THIRTYTWO_RADIO);
    if (outputSize == INDEX_INPUT_NUM) {
        ubOneLoopNum = Align128FloorSize(ubNum / UB_CALCU_BIT_SIXTEEN_RADIO);
    } else if (outputSize == INDEX_INPUT_STOP) {
        ubOneLoopNum = Align128FloorSize(ubNum / UB_CALCU_BIT_EIGHT_RADIO);
    }

    // kernel 切分参数
    int64_t perOfPerCore = (numOfPerCore < ubOneLoopNum) ? numOfPerCore : ubOneLoopNum;
    int64_t loopOfPerCore = Ops::Base::CeilDiv(numOfPerCore, perOfPerCore);
    int64_t tailOfPerCore = numOfPerCore - perOfPerCore * (loopOfPerCore - 1);
    int64_t perOfTailCore = (numOfTailCore < ubOneLoopNum) ? numOfTailCore : ubOneLoopNum;
    int64_t loopOfTailCore = Ops::Base::CeilDiv(numOfTailCore, perOfTailCore);
    int64_t tailOfTailCore = numOfTailCore - perOfTailCore * (loopOfTailCore - 1);

    // 界限核
    int64_t half = 0;
    int64_t halfWayCoreIdx = 0;
    int64_t forwardNumOfHalfWayCore = 1;
    int64_t backwardNumOfHalfWayCore = 0;
    int64_t perLoopNum = 1;
    int64_t loopOfForward = 1;
    int64_t tailOfForward = 1;
    int64_t loopOfBackward = 1;
    int64_t tailOfBackward = 1;
    if (tilingParam.num > 1) {
        half = tilingParam.num / DOUBLE_UB_SIZE;
        int64_t tmpHalfWayCoreIdx = half / numOfPerCore;
        halfWayCoreIdx = (half % numOfPerCore == 0) ? (tmpHalfWayCoreIdx - 1) : tmpHalfWayCoreIdx;
        forwardNumOfHalfWayCore = half - halfWayCoreIdx * numOfPerCore;
        backwardNumOfHalfWayCore = (halfWayCoreIdx == usedCoreNum - 1) ? (numOfTailCore - forwardNumOfHalfWayCore) :
                                                                         (numOfPerCore - forwardNumOfHalfWayCore);
        perLoopNum = (halfWayCoreIdx == usedCoreNum - 1) ? perOfTailCore : perOfPerCore;
        loopOfForward = Ops::Base::CeilDiv(forwardNumOfHalfWayCore, perLoopNum);
        tailOfForward = forwardNumOfHalfWayCore - perLoopNum * (loopOfForward - 1);
        loopOfBackward = Ops::Base::CeilDiv(backwardNumOfHalfWayCore, perLoopNum);
        tailOfBackward = backwardNumOfHalfWayCore - perLoopNum * (loopOfBackward - 1);
    }

    // 赋值给tilingParam
    tilingParam.usedCoreNum = usedCoreNum;
    tilingParam.ubOneLoopNum = ubOneLoopNum;
    tilingParam.numOfPerCore = numOfPerCore;
    tilingParam.loopOfPerCore = loopOfPerCore;
    tilingParam.perOfPerCore = perOfPerCore;
    tilingParam.tailOfPerCore = tailOfPerCore;
    tilingParam.numOfTailCore = numOfTailCore;
    tilingParam.loopOfTailCore = loopOfTailCore;
    tilingParam.perOfTailCore = perOfTailCore;
    tilingParam.tailOfTailCore = tailOfTailCore;
    tilingParam.halfWayCoreIdx = halfWayCoreIdx;
    tilingParam.forwardNumOfHalfWayCore = forwardNumOfHalfWayCore;
    tilingParam.loopOfForward = loopOfForward;
    tilingParam.tailOfForward = tailOfForward;
    tilingParam.loopOfBackward = loopOfBackward;
    tilingParam.tailOfBackward = tailOfBackward;

    return ge::GRAPH_SUCCESS;
}

bool LinSpaceRegbaseTilingClass::IsCapable()
{
    return Ops::Base::IsRegbaseSocVersion(context_);
}

ge::graphStatus LinSpaceRegbaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        npuArch_ = ascendcPlatform.GetCurNpuArch();
        tilingParam_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        tilingParam_.totalUbSize = static_cast<int64_t>(ubSizePlatForm);
    } else {
        auto compileInfoPtr = reinterpret_cast<const LinSpaceCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        npuArch_ = compileInfoPtr->npuArch;
        tilingParam_.totalCoreNum = compileInfoPtr->totalCoreNum;
        tilingParam_.totalUbSize = compileInfoPtr->ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LinSpaceRegbaseTilingClass::GetShapeAttrsInfo()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(INDEX_OUTPUT_OUT));
    outDataType_ = context_->GetOutputDesc(INDEX_OUTPUT_OUT)->GetDataType();
    return ge::GRAPH_SUCCESS;
}

void LinSpaceRegbaseTilingClass::SetStep()
{
    float step(0);
    if (tilingParam_.num > 1) {
        step = (tilingParam_.stop - tilingParam_.start) / static_cast<float>((tilingParam_.num - 1));
    } else if (tilingParam_.num == 1) {
        tilingParam_.stop = tilingParam_.start;
    }
    tilingParam_.step = step;
}

ge::graphStatus LinSpaceRegbaseTilingClass::DoOpTiling()
{
    OP_LOGD(context_, "[LinSpace] LinSpaceTilingForAscendC running begin.");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_, "Context should not be nullptr."), return ge::GRAPH_FAILED);
    auto tensorStart = context_->GetInputTensor(INDEX_INPUT_START);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorStart);
    auto tensorStop = context_->GetInputTensor(INDEX_INPUT_STOP);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorStop);
    auto tensorNum = context_->GetInputTensor(INDEX_INPUT_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorNum);

    auto dtypeStart = tensorStart->GetDataType();
    auto dtypeStop = tensorStop->GetDataType();
    auto dtypeNum = tensorNum->GetDataType();

    auto ret = CheckDtypeIsValid(context_, dtypeStart, dtypeStop, dtypeNum, outDataType_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (dtypeNum == ge::DT_INT32) {
        auto constDataPtr = tensorNum->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, constDataPtr);
        tilingParam_.num = static_cast<int64_t>(*constDataPtr);
    } else {
        auto constDataPtr = tensorNum->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, constDataPtr);
        tilingParam_.num = static_cast<int64_t>(*constDataPtr);
    }

    float start(0);
    float stop(0);
    OP_CHECK_IF(
        GetConstStartOrStop(context_, tensorStart, start, INDEX_INPUT_START) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set tilingData start fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetConstStartOrStop(context_, tensorStop, stop, INDEX_INPUT_STOP) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set tilingData stop fail."), return ge::GRAPH_FAILED);
    tilingParam_.start = start;
    tilingParam_.stop = stop;
    SetStep();
    ret = CalcLinSpaceTilingParam(tilingParam_, outDataType_);
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "[CalcLinSpaceTilingParam] TilingLinSpace fail to caculate tiling param."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LinSpaceRegbaseTilingClass::PostTiling()
{
    tilingData_.set_totalCoreNum(tilingParam_.totalCoreNum);
    tilingData_.set_totalUbSize(tilingParam_.totalUbSize);
    tilingData_.set_num(tilingParam_.num);
    tilingData_.set_usedCoreNum(tilingParam_.usedCoreNum);
    tilingData_.set_ubOneLoopNum(tilingParam_.ubOneLoopNum);
    tilingData_.set_numOfPerCore(tilingParam_.numOfPerCore);
    tilingData_.set_loopOfPerCore(tilingParam_.loopOfPerCore);
    tilingData_.set_perOfPerCore(tilingParam_.perOfPerCore);
    tilingData_.set_tailOfPerCore(tilingParam_.tailOfPerCore);
    tilingData_.set_numOfTailCore(tilingParam_.numOfTailCore);
    tilingData_.set_loopOfTailCore(tilingParam_.loopOfTailCore);
    tilingData_.set_perOfTailCore(tilingParam_.perOfTailCore);
    tilingData_.set_tailOfTailCore(tilingParam_.tailOfTailCore);
    tilingData_.set_halfWayCoreIdx(tilingParam_.halfWayCoreIdx);
    tilingData_.set_forwardNumOfHalfWayCore(tilingParam_.forwardNumOfHalfWayCore);
    tilingData_.set_loopOfForward(tilingParam_.loopOfForward);
    tilingData_.set_tailOfForward(tilingParam_.tailOfForward);
    tilingData_.set_loopOfBackward(tilingParam_.loopOfBackward);
    tilingData_.set_tailOfBackward(tilingParam_.tailOfBackward);
    tilingData_.set_start(tilingParam_.start);
    tilingData_.set_stop(tilingParam_.stop);
    tilingData_.set_step(tilingParam_.step);

    // 设置 userWorkspace
    size_t* userWorkspaceSize = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;

    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingDataPtr);
    if (tilingData_.GetDataSize() > rawTilingDataPtr->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(rawTilingDataPtr->GetData(), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(tilingData_.GetDataSize());

    context_->SetBlockDim(tilingData_.get_usedCoreNum());
    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

uint64_t LinSpaceRegbaseTilingClass::GetTilingKey() const
{
    return DOUBLE_CAST_TILING_KEY;
}

REGISTER_OPS_TILING_TEMPLATE(LinSpace, LinSpaceRegbaseTilingClass, 3000);

} // namespace optiling