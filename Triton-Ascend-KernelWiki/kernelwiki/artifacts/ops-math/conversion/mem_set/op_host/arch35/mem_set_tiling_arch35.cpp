/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mem_set_tiling_arch35.cpp
 * \brief tiling for mem set
 */

#include <set>
#include <string>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "mem_set_tiling_arch35.h"
#include "op_host/tiling_base_util.h"

using namespace MemSetTpl;
namespace optiling {
std::set<ge::DataType> SUPPORT_TYPE_LIST = {ge::DT_INT8,   ge::DT_INT32, ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,
                                            ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16};
std::set<ge::DataType> SUPPORT_TYPE_INT_LIST = {ge::DT_INT8,   ge::DT_INT32,  ge::DT_UINT8, ge::DT_INT16,
                                                ge::DT_UINT16, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};
constexpr uint64_t BLOCK_SIZE = 512;

template <uint16_t Count>
void MemSetTilingClass::CheckTilingData(MemSetTilingData<Count>* tilingDataPostAtr)
{
    for (int32_t i = 0; i < inputCount_; i++) {
        OP_LOGD(context_, "intValue %ld", tilingDataPostAtr->intValue[i]);
        OP_LOGD(context_, "floatValue %f", tilingDataPostAtr->floatValue[i]);
        OP_LOGD(context_, "perCoreSizes %ld", tilingDataPostAtr->perCoreSizes[i]);
        OP_LOGD(context_, "lastCoreSizes %ld", tilingDataPostAtr->lastCoreSizes[i]);
        OP_LOGD(context_, "listType %u", tilingDataPostAtr->listType[i]);
        OP_LOGD(context_, "useCore %u", tilingDataPostAtr->useCore[i]);
    }
    OP_LOGD(context_, "halfUbSize %d", tilingDataPostAtr->halfUbSize);
    OP_LOGD(context_, "inputCount %u", tilingDataPostAtr->inputCount);
}

uint64_t MemSetTilingClass::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TilingKey_);
    return tilingKey;
}

template <uint16_t Count>
void MemSetTilingClass::PostDo()
{
    TilingKey_ = Count;
    auto* tilingDataPostAtr = context_->GetTilingData<MemSetTilingData<Count>>();
    for (int32_t i = 0; i < inputCount_; i++) {
        tilingDataPostAtr->intValue[i] = intValue_[i];
        tilingDataPostAtr->floatValue[i] = floatValue_[i];
        tilingDataPostAtr->lastCoreSizes[i] = lastCoreSizes_[i];
        tilingDataPostAtr->perCoreSizes[i] = perCoreSizes_[i];
        tilingDataPostAtr->listType[i] = listType_[i];
        tilingDataPostAtr->useCore[i] = useCore_[i];
    }
    tilingDataPostAtr->halfUbSize = halfUbSize_;
    tilingDataPostAtr->inputCount = inputCount_;
    CheckTilingData<Count>(tilingDataPostAtr);
}

ge::graphStatus MemSetTilingClass::PostTiling()
{
    // Different sizes of tiling data templates
    const std::vector<int> validNums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 192, 256};
    if (inputCount_ > validNums.back()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "TensorNum",
            std::to_string(inputCount_).c_str(),
            "The value of TensorNum must be within the range [1, 256].");
        return ge::GRAPH_FAILED;
    }
    auto it = std::lower_bound(validNums.begin(), validNums.end(), inputCount_);
    int targetNum = *it;
    switch (targetNum) {
        case 1: PostDo<1>();break;
        case 2: PostDo<2>();break;
        case 3: PostDo<3>();break;
        case 4: PostDo<4>();break;
        case 5: PostDo<5>();break;
        case 6: PostDo<6>();break;
        case 7: PostDo<7>();break;
        case 8: PostDo<8>();break;
        case 9: PostDo<9>();break;
        case 10: PostDo<10>();break;
        case 11: PostDo<11>();break;
        case 12: PostDo<12>();break;
        case 13: PostDo<13>();break;
        case 14: PostDo<14>();break;
        case 15: PostDo<15>();break;
        case 16: PostDo<16>();break;
        case 32: PostDo<32>();break;
        case 64: PostDo<64>();break;
        case 128: PostDo<128>();break;
        case 196: PostDo<196>();break;
        case 256: PostDo<256>();break;
    }
    context_->SetBlockDim(aicoreParams_.numBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MemSetTilingClass::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MemSetTilingClass::DoOpTiling()
{
    for (uint16_t i = 0; i < inputCount_; i++) {
        uint32_t dataSize = ge::GetSizeByDataType(static_cast<ge::DataType>(listType_[i]));
        if (dataSize == 0) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "listType",
                ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(listType_[i])).c_str(),
                "The dtype of input must be within the range [INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16].");
            return ge::GRAPH_FAILED;
        }
        uint64_t sizeByte = sizes_[i];
        if (sizeByte == 0ULL) {
            OP_LOGW(context_, "Input[%d] has zero size, skip tiling logic", i);
            perCoreSizes_[i] = 0;
            lastCoreSizes_[i] = 0;
            useCore_[i] = 0;
            continue;
        }
        uint32_t tail = sizeByte % BLOCK_SIZE;
        uint16_t tailOrNot = tail > 0 ? 1 : 0;
        uint64_t blockNum = sizeByte / BLOCK_SIZE + tailOrNot;
        perCoreSizes_[i] = (blockNum + aicoreParams_.numBlocks - 1) / aicoreParams_.numBlocks;
        useCore_[i] = (blockNum + perCoreSizes_[i] - 1) / perCoreSizes_[i];
        lastCoreSizes_[i] = blockNum - (useCore_[i] - 1) * perCoreSizes_[i];
        perCoreSizes_[i] = BLOCK_SIZE * perCoreSizes_[i] / dataSize;
        lastCoreSizes_[i] = ((lastCoreSizes_[i] - tailOrNot) * BLOCK_SIZE + tail) / dataSize;
        sizes_[i] = sizes_[i] / dataSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MemSetTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (isDynamic_ || platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MemSetCompileInfoArch35>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        aicoreParams_.numBlocks = compileInfoPtr->coreNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    }
    if (aicoreParams_.numBlocks == 0LL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "numBlocks", "0",
            "The value of numBlocks must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    if (aicoreParams_.ubSize == 0LL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize", "0",
            "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    halfUbSize_ = aicoreParams_.ubSize / 2;
    if (cacheLineSize_ == 0LL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cacheLineSize", "0",
            "The value of cacheLineSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MemSetTilingClass::AllocTilingStruct()
{
    perCoreSizes_.resize(inputCount_);
    lastCoreSizes_.resize(inputCount_);
    intValue_.resize(inputCount_);
    floatValue_.resize(inputCount_);
    listType_.resize(inputCount_);
    useCore_.resize(inputCount_);
    sizes_.resize(inputCount_);
}

ge::graphStatus MemSetTilingClass::SetShapeAttrsInfo(bool isGE)
{
    const auto* attrs = context_->GetAttrs();
    const auto* sizesPtr = attrs->GetListInt(0);
    const auto* dataTypesPtr = attrs->GetListInt(1);
    const auto* valueIntPtr = attrs->GetListInt(2);
    const auto* valueFloatPtr = attrs->GetListFloat(3);

    uint16_t valueIntIndex = 0;
    uint16_t valueFloatIndex = 0;
    for (uint16_t i = 0; i < inputCount_; i++) {
        ge::DataType dtype = static_cast<ge::DataType>(dataTypesPtr->GetData()[i]);
        if (SUPPORT_TYPE_LIST.find(dtype) == SUPPORT_TYPE_LIST.end()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dataTypes",
                ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
                "The dtype of dataTypes must be within the range [INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16].");
            return ge::GRAPH_FAILED;
        }
        listType_[i] = static_cast<uint16_t>(dtype);
        if (isGE) {
            if (SUPPORT_TYPE_INT_LIST.find(dtype) != SUPPORT_TYPE_INT_LIST.end()) {
                intValue_[i] = valueIntPtr->GetData()[valueIntIndex];
                floatValue_[i] = 0.0f;
                valueIntIndex++;
            } else {
                floatValue_[i] = valueFloatPtr->GetData()[valueFloatIndex];
                intValue_[i] = 0;
                valueFloatIndex++;
            }
        } else {
            floatValue_[i] = valueFloatPtr->GetData()[i * 2]; //input is 64B stred, read as float32, hence*2
            intValue_[i] = valueIntPtr->GetData()[i];
        }
        if (isDynamic_) {
            auto memSetContext = static_cast<ops::AtomicCleanTilingContext*>(context_);
            sizes_[i] = memSetContext->GetCleanOutputSize(i);
        } else {
            sizes_[i] = sizesPtr->GetData()[i];
        }
        if (sizes_[i] % ge::GetSizeByDataType(static_cast<ge::DataType>(listType_[i])) != 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "sizes and dataType",
                (std::to_string(sizes_[i]) + " and " +
                    ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(listType_[i]))).c_str(),
                "The value of sizes must be exactly divisible by dataType size.");
            return ge::GRAPH_FAILED;
        }
        OP_LOGI(context_, "dynamic is [%d] sizes[%u] is %ld", static_cast<int>(isDynamic_), i, sizes_[i]);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MemSetTilingClass::GetShapeAttrsInfo()
{
    const auto* attrs = context_->GetAttrs();
    const auto* sizesPtr = attrs->GetListInt(0);
    const auto* dataTypesPtr = attrs->GetListInt(1);
    const auto* valueIntPtr = attrs->GetListInt(2);
    const auto* valueFloatPtr = attrs->GetListFloat(3);
    if (sizesPtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "sizes", "nullptr", "sizes cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (dataTypesPtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dataTypes", "nullptr", "dataTypes cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (valueIntPtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valueInt", "nullptr", "valueInt cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (valueFloatPtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valueFloat", "nullptr", "valueFloat cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    uint16_t sizesDim = sizesPtr->GetCapacity();
    uint16_t dataTypesDim = dataTypesPtr->GetCapacity();
    uint16_t valueIntDim = valueIntPtr->GetCapacity();
    uint16_t valueFloatDim = valueFloatPtr->GetCapacity();
    OP_LOGD(
        context_->GetNodeName(), "sizesdim %d and typedim %d and valuesint %d and valueFloatDim %d", sizesDim,
        dataTypesDim, valueIntDim, valueFloatDim);
    bool isGE = false;
    if (sizesDim != dataTypesDim || sizesDim != valueIntDim || sizesDim != valueFloatDim) {
        OP_LOGW(context_, "sizesdim and typedim and valuesint and valueFloatDim is Not equal");
        isGE = true;
    }
    inputCount_ = sizesDim;
    if (sizesPtr->GetData()[0] <= 0 || inputCount_ == 0) {
        isDynamic_ = true;
    }
    if (isDynamic_) {
        inputCount_ = context_->GetComputeNodeInfo()->GetInputsNum() - 1;
    }
    OP_LOGD(context_, "dim is %u", inputCount_);
    AllocTilingStruct();
    if (SetShapeAttrsInfo(isGE) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4MemSetArch35(gert::TilingContext* context)
{
    OP_LOGI("MemSet tilingData", "Start tiling for MemSet.");
    const MemSetCompileInfoArch35* compileInfo =
        context->GetCompileInfo<MemSetCompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "runing regbase soc version tiling func");
    class MemSetTilingClass tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4MemSetArch35(gert::TilingParseContext* context)
{
    OP_LOGI("MemSet tilingData", "Start tiling prepare for MemSet.");
    auto compileInfo = context->GetCompiledInfo<MemSetCompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", "0",
            "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    compileInfo->ubSize = ubSize;
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", "0",
            "The value of coreNum must be greater than 0.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu.", compileInfo->coreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MemSet).Tiling(Tiling4MemSetArch35).TilingParse<MemSetCompileInfoArch35>(TilingPrepare4MemSetArch35);
} // namespace optiling