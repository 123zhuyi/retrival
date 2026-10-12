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
 * \file scatter_pa_kv_cache_tiling_arch35.cpp
 * \brief
 */

#include <algorithm>
#include "scatter_pa_kv_cache_tiling.h"
#include "tiling/tiling_api.h"

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
#include "op_common/op_host/util/platform_util.h"


namespace optiling {
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIM3 = 3;
constexpr int64_t DIM4 = 4;

constexpr int64_t INDEX_INPUT_KEY = 0;
constexpr int64_t INDEX_INPUT_KEY_CACHE_IN = 1;
constexpr int64_t INDEX_SLOT_MAPPING = 2;
constexpr int64_t INDEX_COMPRESS_LENS = 3;
constexpr int64_t INDEX_COMPRESS_SEQ_OFFSET = 4;
constexpr int64_t INDEX_INPUT_SEQ_LENS = 5;
constexpr int64_t INDEX_INPUT_VALUE = 3;
constexpr int64_t INDEX_INPUT_VALUE_CACHE_IN = 4;
constexpr int64_t DUAL_IN_OUT_MODE_INDEX_OFFSET = 2;

constexpr uint64_t HUNDRED = 100;
constexpr uint64_t TEN = 10;

constexpr int64_t FULLY_LOAD = 1;
constexpr int64_t NOT_FULLY_LOAD = 0;

constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t INT64_DTYPE_SIZE = 8;

constexpr int64_t TEMPLATE_NORMAL = 1;
constexpr int64_t TEMPLATE_ROPE = 2;
constexpr int64_t TEMPLATE_ALIBI = 3;

constexpr int64_t SINGLE_IN_OUT = 1;
constexpr int64_t DUAL_IN_OUT = 2;

constexpr int64_t BLOCK_SIZE = 32;

constexpr int64_t MAX_HANLDE_BYTE_SIZE_PER_LOOP = 16384;

constexpr int64_t RESERVED_UB_SIZE = static_cast<int64_t>(8 * 1024);
constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;

bool ScatterPaKvCacheTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterPaKvCacheTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum_ == 0) {
        OP_LOGE(context_, "totalCoreNum is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE(context_, "ubSize is 0");
        return ge::GRAPH_FAILED;
    }
    ubSize_ = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckSlotMappingShape(int64_t requiredDimNum)
{
    auto inputSlotMapping = context_->GetRequiredInputTensor(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMapping);
    auto inputSlotMappingShape = inputSlotMapping->GetStorageShape();
    size_t inputSlotMappingDimNum = inputSlotMappingShape.GetDimNum();
    if (inputSlotMappingDimNum != static_cast<size_t>(requiredDimNum)) {
        OP_LOGE(context_, "slot_mapping dim num must be %ld", requiredDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetIndexDtype()
{
    auto inputSlotMappingDesc = context_->GetInputDesc(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMappingDesc);
    ge::DataType inputSlotMappingDtype = inputSlotMappingDesc->GetDataType();
    if (inputSlotMappingDtype == ge::DT_INT32) {
        indexDtypeSize_ = INT32_DTYPE_SIZE;
    } else if (inputSlotMappingDtype == ge::DT_INT64) {
        indexDtypeSize_ = INT64_DTYPE_SIZE;
    } else {
        OP_LOGE(context_, "slot_mapping dtype must be int32 or int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetInputDtype()
{
    auto inputKeyDesc = context_->GetInputDesc(inputKey_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyDesc);
    ge::DataType inputKeyDtype = inputKeyDesc->GetDataType();

    auto inputKeyCacheInDesc = context_->GetInputDesc(inputKeyCacheIn_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyCacheInDesc);
    ge::DataType inputKeyCacheInDtype = inputKeyCacheInDesc->GetDataType();

    if (inOutMode_ == DUAL_IN_OUT) {
        auto inputValueDesc = context_->GetInputDesc(inputValue_);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
        ge::DataType inputValueDtype = inputValueDesc->GetDataType();
        auto inputValueCacheInDesc = context_->GetInputDesc(inputValueCacheIn_);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueCacheInDesc);
        ge::DataType inputValueCacheInDtype = inputValueCacheInDesc->GetDataType();
        bool inputKeyDtypeFailCheck = inputKeyDtype != inputValueDtype || inputKeyDtype != inputKeyCacheInDtype;
        inputKeyDtypeFailCheck = inputKeyDtypeFailCheck || inputKeyDtype != inputValueCacheInDtype;
        OP_CHECK_IF(inputKeyDtypeFailCheck, OP_LOGE(context_, "key, value, key_cache, value_cache dtype must be same"),
                    return ge::GRAPH_FAILED;);
    } else if (inOutMode_ == SINGLE_IN_OUT) {
        OP_CHECK_IF(inputKeyDtype != inputKeyCacheInDtype, OP_LOGE(context_, "key and key_cache dtype must be same"),
                    return ge::GRAPH_FAILED;);
    }

    inputDtype_ = inputKeyDtype;
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        dtypeByteSize_ = DIM1;
    } else {
        dtypeByteSize_ = static_cast<int64_t>(GetSizeByDataType(inputDtype_));
    }
    OP_CHECK_IF(dtypeByteSize_ <= 0, OP_LOGE(context_, "get input dtype bytes failed."), return ge::GRAPH_FAILED;);
    bool inputDtypeCheck = inputDtype_ == ge::DT_FLOAT || inputDtype_ == ge::DT_FLOAT16 || inputDtype_ == ge::DT_BF16 ||
                           inputDtype_ == ge::DT_INT8 || inputDtype_ == ge::DT_UINT8 || inputDtype_ == ge::DT_INT16 ||
                           inputDtype_ == ge::DT_UINT16 || inputDtype_ == ge::DT_INT32 ||
                           inputDtype_ == ge::DT_UINT32 || inputDtype_ == ge::DT_HIFLOAT8 ||
                           inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
                           inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2;
    OP_CHECK_IF(!inputDtypeCheck, OP_LOGE(context_, "input dtype not support."), return ge::GRAPH_FAILED;);
    return ge::GRAPH_SUCCESS;
}

int64_t ScatterPaKvCacheTiling::RoundUp(int64_t x, int64_t dtypeSize)
{
    // 入参保证 UbBlockSize 和 dtypeSize 不为0
    OP_CHECK_IF(dtypeSize == 0, OP_LOGD(this->context_, "dtypeSize is 0."), return x);
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(this->context_));
    OP_CHECK_IF(ubBlockSize == 0, OP_LOGD(this->context_, "UbBlockSize is 0."), return 0);
    int64_t elemNum = ubBlockSize / dtypeSize;
    return (x + elemNum - 1) / elemNum * elemNum;
}

void ScatterPaKvCacheTiling::GetCommonTilingInfo()
{
    numTokens_ = inputKeyShape_.GetDim(DIM0) * inputKeyShape_.GetDim(DIM2);
    blockFactor_ = Ops::Base::CeilDiv<int64_t>(numTokens_, totalCoreNum_);
    usedCoreNum_ = std::min(Ops::Base::CeilDiv<int64_t>(numTokens_, blockFactor_), totalCoreNum_);
    tailBlockFactor_ = numTokens_ - blockFactor_ * (usedCoreNum_ - 1);
    kHeadSize_ = inputKeyShape_.GetDim(DIM3);
    keyStride0_ = inputKeyShape_.GetDim(DIM1) * inputKeyShape_.GetDim(DIM2) * inputKeyShape_.GetDim(DIM3);
    keyStride1_ = inputKeyShape_.GetDim(DIM2) * inputKeyShape_.GetDim(DIM3);
    keyStride2_ = inputKeyShape_.GetDim(DIM3);
    if (inOutMode_ == SINGLE_IN_OUT) {
        vHeadSize_ = DIM0;
        valueStride0_ = DIM0;
        valueStride1_ = DIM0;
        valueStride2_ = DIM0;
    } else if (inOutMode_ == DUAL_IN_OUT) {
        vHeadSize_ = inputValueShape_.GetDim(DIM3);
        valueStride0_ = inputValueShape_.GetDim(DIM1) * inputValueShape_.GetDim(DIM2) * inputValueShape_.GetDim(DIM3);
        valueStride1_ = inputValueShape_.GetDim(DIM2) * inputValueShape_.GetDim(DIM3);
        valueStride2_ = inputValueShape_.GetDim(DIM3);
    }
    batch_ = inputKeyShape_.GetDim(DIM0);
    seqLen_ = inputKeyShape_.GetDim(DIM1);
    numHead_ = inputKeyShape_.GetDim(DIM2);
    numBlocks_ = inputKeyCacheInShape_.GetDim(DIM0);
    blockSize_ = inputKeyCacheInShape_.GetDim(DIM1);
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateNormal()
{
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    numBlocks_ = inputKeyCacheInShape_.GetDim(DIM0);
    blockSize_ = inputKeyCacheInShape_.GetDim(DIM1);
    numTokens_ = inputKeyShape_.GetDim(DIM0);
    kHandleNumPerCore_ = inputKeyShape_.GetDim(DIM1) * inputKeyShape_.GetDim(DIM2);
    vHandleNumPerCore_ =
        (inOutMode_ == SINGLE_IN_OUT) ? DIM0 : inputValueShape_.GetDim(DIM1) * inputValueShape_.GetDim(DIM2);
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(inputKeyShape_.GetDim(DIM2) % DIM2 != 0,
                    OP_LOGE(context_, "k_head_size must be an even number when input dtype is fp4."),
                    return ge::GRAPH_FAILED;);
        OP_CHECK_IF(inOutMode_ == DUAL_IN_OUT && inputValueShape_.GetDim(DIM2) % DIM2 != 0,
                    OP_LOGE(context_, "v_head_size must be an even number when input dtype is fp4."),
                    return ge::GRAPH_FAILED;);
        kHandleNumPerCore_ /= DIM2;
        vHandleNumPerCore_ /= DIM2;
    }
    blockFactor_ = Ops::Base::CeilDiv<int64_t>(numTokens_, totalCoreNum_);
    usedCoreNum_ = std::min(Ops::Base::CeilDiv<int64_t>(numTokens_, blockFactor_), totalCoreNum_);
    tailBlockFactor_ = numTokens_ - blockFactor_ * (usedCoreNum_ - 1);
    int64_t maxHandleNumPerLoop = ubSize_ / dtypeByteSize_;
    // check whethere tail dim can fully load.
    int64_t ubThreshold = std::max(blockFactor_, tailBlockFactor_) *
                              (RoundUp(kHandleNumPerCore_, dtypeByteSize_) +
                               RoundUp(vHandleNumPerCore_, dtypeByteSize_)) + // inputKey & inputValue
                          RoundUp(std::max(blockFactor_, tailBlockFactor_), dtypeByteSize_) * DIM2 * indexDtypeSize_ /
                              dtypeByteSize_; // slotMapping for key and value
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    if (CheckSlotMappingShape(DIM1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // tail dim can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;
    kHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHandleNumPerCore_, kHandleNumPerLoop_);
    kTailHandleNum_ = kHandleNumPerCore_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
    kLoopNum_--;
    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHandleNumPerCore_, vHandleNumPerLoop_);
        vTailHandleNum_ = vHandleNumPerCore_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
        vLoopNum_--;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateRope()
{
    if (inOutMode_ == SINGLE_IN_OUT &&
        (inputDtype_ == ge::DT_HIFLOAT8 || inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
         inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2)) {
        OP_LOGE(context_, "input dtype not support in rope compression when single input and output.");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetCommonTilingInfo();
    // check whethere tail dim can fully load.
    int64_t compressSeqOffsetSize = inputKeyShape_.GetDim(DIM0) * inputKeyShape_.GetDim(DIM2);
    int64_t alignKHead = RoundUp(kHeadSize_, dtypeByteSize_);
    int64_t alignVHead = RoundUp(vHeadSize_, dtypeByteSize_);
    int64_t numKHeadSize = seqLen_ * alignKHead;
    int64_t numVHeadSize = seqLen_ * alignVHead;
    int64_t maxHandleNumPerLoop = ubSize_ / dtypeByteSize_;
    int64_t floatFactor = (INT32_DTYPE_SIZE / dtypeByteSize_);
    int64_t inOutModeDim = (inOutMode_ == SINGLE_IN_OUT) ? DIM1 : DIM2;
    int64_t ubThreshold =
        std::max(numKHeadSize, numVHeadSize) * inOutModeDim +      // for inputKeyLocal & inputValueLocal
        blockFactor_ * DIM1 +                                      // slotMapping for key & value
        blockFactor_ * DIM1 +                                      // seqLens
        blockFactor_ * DIM1 +                                      // compressLen
        compressSeqOffsetSize * indexDtypeSize_ / dtypeByteSize_ + // compress_seq_offset size
        std::max(alignKHead, alignVHead) * floatFactor * DIM1 +    // reduce Buf for inputKeyLocal or inputValueLocal
        std::max(alignKHead, alignVHead) * floatFactor * DIM1 +    // divide Buf
        std::max(alignKHead, alignVHead) * floatFactor * DIM1;     // cast Buf
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;

    kHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHeadSize_, kHandleNumPerLoop_);
    kTailHandleNum_ = kHeadSize_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
    kLoopNum_--;

    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHeadSize_, vHandleNumPerLoop_);
        vTailHandleNum_ = vHeadSize_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
        vLoopNum_--;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateAlibi()
{
    if (inOutMode_ == SINGLE_IN_OUT &&
        (inputDtype_ == ge::DT_HIFLOAT8 || inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
         inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2)) {
        OP_LOGE(context_, "input dtype not support in alibi compression when single input and output.");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetCommonTilingInfo();
    // check whethere tail dim can fully load.
    int64_t maxHandleNumPerLoop = ubSize_ / dtypeByteSize_;
    int64_t inOutModeDim = (inOutMode_ == SINGLE_IN_OUT) ? DIM1 : DIM2;
    int64_t ubThreshold = RoundUp(std::max(kHeadSize_, vHeadSize_), dtypeByteSize_) * seqLen_ *
                          inOutModeDim +                                        // for inputKeyLocal & inputValueLocal
                          blockFactor_ * DIM1 +                                 // slotMapping for key & value
                          blockFactor_ * DIM1;                                  // compressLen
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;

    kHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHeadSize_, kHandleNumPerLoop_);
    kTailHandleNum_ = kHeadSize_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
    kLoopNum_--;

    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHeadSize_, vHandleNumPerLoop_);
        vTailHandleNum_ = vHeadSize_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
        vLoopNum_--;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckNormal()
{
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim 0 of key and slot_mapping must be equal. "), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM1) != inputKeyCacheInShape_.GetDim(DIM2),
                OP_LOGE(context_, "the dim 1 of key must be equal to the dim 2 of key_cache."),
                return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM2) != inputKeyCacheInShape_.GetDim(DIM3),
                OP_LOGE(context_, "the dim2 of key must be equal to the dim3 of key_cache."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inOutMode_ != DUAL_IN_OUT, , return ge::GRAPH_SUCCESS;); // 卫语句
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim 0 of key and value must be equal. "), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(
        inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1) ||
            inputKeyShape_.GetDim(DIM1) != inputValueCacheInShape_.GetDim(DIM2),
        OP_LOGE(context_, "the dim 1 of key and value must be equal to the dim 2 of key_cache and value_cache."),
        return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputValueShape_.GetDim(DIM2) != inputValueCacheInShape_.GetDim(DIM3),
                OP_LOGE(context_, "the dim2 of value must be equal to the dim3 of value_cache."),
                return ge::GRAPH_FAILED;);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckRope()
{
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0) ||
                    inputKeyShape_.GetDim(DIM0) != compressLensShape_.GetDim(DIM0) ||
                    inputKeyShape_.GetDim(DIM0) != seqLensShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim 0 of key, slot_mapping, compress_lens and seq_lens must be equal."),
                return ge::GRAPH_FAILED;);
    OP_CHECK_IF(
        inputKeyShape_.GetDim(DIM2) != slotMappingShape_.GetDim(DIM1) ||
            inputKeyShape_.GetDim(DIM2) != compressLensShape_.GetDim(DIM1),
        OP_LOGE(context_,
                "the dim2 of the key must be equal to the dim 1 of slot_mapping and the dim 1 of compress_lens."),
        return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM3) != inputKeyCacheInShape_.GetDim(DIM3),
                OP_LOGE(context_, "the dim3 of key must be equal to the dim3 of key_cache."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyCacheInShape_.GetDim(DIM2) != DIM1,
                OP_LOGE(context_, "the dim2 of key_cache must be equal to 1."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(compressLensShape_.GetDim(DIM0) * compressLensShape_.GetDim(DIM1) !=
                    compressSeqOffsetShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim0 of compress_seq_offset must be equal to the dim 0 of compress_lens "
                                  "multiplied by the dim 1 of the compress_lens."),
                return ge::GRAPH_FAILED;);

    OP_CHECK_IF(inOutMode_ != DUAL_IN_OUT, , return ge::GRAPH_SUCCESS;); // 卫语句
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim 0 of key and value must be equal."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1),
                OP_LOGE(context_, "the dim1 of the key and value must be equal."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM2) != inputValueShape_.GetDim(DIM2),
                OP_LOGE(context_, "the dim2 of the key must be equal to the dim2 of value."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputValueShape_.GetDim(DIM3) != inputValueCacheInShape_.GetDim(DIM3),
                OP_LOGE(context_, "the dim3 of value must be equal to the dim3 of value_cache."),
                return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputValueCacheInShape_.GetDim(DIM2) != DIM1,
                OP_LOGE(context_, "the dim2 of value_cache must be equal to 1."), return ge::GRAPH_FAILED;);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckAlibi()
{
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0) ||
                    inputKeyShape_.GetDim(DIM0) != seqLensShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim 0 of key, slot_mapping and seq_lens must be equal."),
                return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyCacheInShape_.GetDim(DIM2) != DIM1,
                OP_LOGE(context_, "the dim2 of key_cache must be equal to 1."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM2) != slotMappingShape_.GetDim(DIM1),
                OP_LOGE(context_, "the dim2 of key must be equal to the dim1 of slot_mapping."),
                return ge::GRAPH_FAILED;);

    OP_CHECK_IF(inOutMode_ != DUAL_IN_OUT, , return ge::GRAPH_SUCCESS;); // 卫语句
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0),
                OP_LOGE(context_, "the dim 0 of key and value must be equal."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1),
                OP_LOGE(context_, "the dim1 of key and value must be equal."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputValueCacheInShape_.GetDim(DIM2) != DIM1,
                OP_LOGE(context_, "the dim2 of value_cache must be equal to 1."), return ge::GRAPH_FAILED;);
    OP_CHECK_IF(inputKeyShape_.GetDim(DIM2) != inputValueShape_.GetDim(DIM2),
                OP_LOGE(context_, "the dim2 of key and value must be equal."), return ge::GRAPH_FAILED;);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckDimValid()
{
    if (templateType_ == TEMPLATE_NORMAL) {
        return CheckNormal();
    }
    if (templateType_ == TEMPLATE_ROPE) {
        return CheckRope();
    }
    if (templateType_ == TEMPLATE_ALIBI) {
        return CheckAlibi();
    }
    OP_LOGE(context_, "input is not supproted, please check input.");
    return ge::GRAPH_FAILED;
}

void ScatterPaKvCacheTiling::SetInputPos()
{
    if (inOutMode_ == SINGLE_IN_OUT) {
        inputKey_ = INDEX_INPUT_KEY;
        inputKeyCacheIn_ = INDEX_INPUT_KEY_CACHE_IN;
        inputSlotMapping_ = INDEX_SLOT_MAPPING;
        inputCompressLens_ = INDEX_COMPRESS_LENS;
        inputCompressSeqOffset_ = INDEX_COMPRESS_SEQ_OFFSET;
        inputSeqLens_ = INDEX_INPUT_SEQ_LENS;
    } else if (inOutMode_ == DUAL_IN_OUT) {
        inputKey_ = INDEX_INPUT_KEY;
        inputKeyCacheIn_ = INDEX_INPUT_KEY_CACHE_IN;
        inputSlotMapping_ = INDEX_SLOT_MAPPING;
        inputValue_ = INDEX_INPUT_VALUE;
        inputValueCacheIn_ = INDEX_INPUT_VALUE_CACHE_IN;
        inputCompressLens_ = INDEX_COMPRESS_LENS + DUAL_IN_OUT_MODE_INDEX_OFFSET;
        inputCompressSeqOffset_ = INDEX_COMPRESS_SEQ_OFFSET + DUAL_IN_OUT_MODE_INDEX_OFFSET;
        inputSeqLens_ = INDEX_INPUT_SEQ_LENS + DUAL_IN_OUT_MODE_INDEX_OFFSET;
    }
}

ge::graphStatus ScatterPaKvCacheTiling::GetShapeAttrsInfo()
{
    SetInputPos();
    OP_CHECK_IF(GetInputDtype() != ge::GRAPH_SUCCESS || GetIndexDtype() != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED;);
    auto inputKey = context_->GetRequiredInputTensor(inputKey_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKey);
    inputKeyShape_ = inputKey->GetStorageShape();
    size_t inputKeyDimNum = inputKeyShape_.GetDimNum();

    auto inputKeyCacheIn = context_->GetRequiredInputTensor(inputKeyCacheIn_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyCacheIn);
    inputKeyCacheInShape_ = inputKeyCacheIn->GetStorageShape();

    auto slotMapping = context_->GetRequiredInputTensor(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, slotMapping);
    slotMappingShape_ = slotMapping->GetStorageShape();

    auto inputValue = context_->GetRequiredInputTensor(inputValue_);
    auto inputValueCacheIn = context_->GetRequiredInputTensor(inputValueCacheIn_);
    if (inOutMode_ == DUAL_IN_OUT) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValue);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueCacheIn);
        inputValueShape_ = inputValue->GetStorageShape();
        inputValueCacheInShape_ = inputValueCacheIn->GetStorageShape();
        size_t inputValueDimNum = inputValueShape_.GetDimNum();
        OP_CHECK_IF(inputKeyDimNum != inputValueDimNum,
                    OP_LOGE(context_, "the dim num of inputKey and inputValue are not the same."),
                    return ge::GRAPH_FAILED;);
    }
    OP_CHECK_IF(inputKeyDimNum != static_cast<size_t>(DIM3) && inputKeyDimNum != static_cast<size_t>(DIM4),
                OP_LOGE(context_, "the dim num of inputKey must be 3 or 4."), return ge::GRAPH_FAILED;);
    // entering template normal
    OP_CHECK_IF(inputKeyDimNum == static_cast<size_t>(DIM3), OP_LOGI(context_, "the dim num of inputKey is 3."),
                templateType_ = TEMPLATE_NORMAL;
                return ge::GRAPH_SUCCESS;);
    // else: inputKeyDimNum is 4
    auto compressLens = context_->GetOptionalInputTensor(inputCompressLens_);
    auto compressSeqOffset = context_->GetOptionalInputTensor(inputCompressSeqOffset_);
    auto seqLens = context_->GetOptionalInputTensor(inputSeqLens_);
    if (compressLens != nullptr && compressSeqOffset != nullptr && seqLens != nullptr) {
        // entering template rope
        compressLensShape_ = compressLens->GetStorageShape();
        compressSeqOffsetShape_ = compressSeqOffset->GetStorageShape();
        seqLensShape_ = seqLens->GetStorageShape();
        templateType_ = TEMPLATE_ROPE;
    } else if (compressLens != nullptr && compressSeqOffset == nullptr && seqLens != nullptr) {
        // entering template alibi
        compressLensShape_ = compressLens->GetStorageShape();
        seqLensShape_ = seqLens->GetStorageShape();
        templateType_ = TEMPLATE_ALIBI;
    } else {
        OP_LOGE(context_, "when dim num of inputKey is 4, compress_lens and seq_lens must not be None.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::DoOpTiling()
{
    if (templateType_ == TEMPLATE_NORMAL) {
        return TemplateNormal();
    } else if (templateType_ == TEMPLATE_ROPE) {
        return TemplateRope();
    } else if (templateType_ == TEMPLATE_ALIBI) {
        return TemplateAlibi();
    }
    return ge::GRAPH_FAILED;
}

uint64_t ScatterPaKvCacheTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ScatterPaKvCacheTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

void ScatterPaKvCacheTiling::GenTilingKey()
{
    tilingKey_ = static_cast<uint64_t>(templateType_) * HUNDRED + static_cast<uint64_t>(indexDtypeSize_) * TEN +
                 static_cast<uint64_t>(isFullyLoad_);
}

ge::graphStatus ScatterPaKvCacheTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(usedCoreNum_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_kHandleNumPerCore(kHandleNumPerCore_);
    tilingData_.set_vHandleNumPerCore(vHandleNumPerCore_);
    tilingData_.set_kLoopNum(kLoopNum_);
    tilingData_.set_vLoopNum(vLoopNum_);
    tilingData_.set_kHandleNumPerLoop(kHandleNumPerLoop_);
    tilingData_.set_vHandleNumPerLoop(vHandleNumPerLoop_);
    tilingData_.set_kTailHandleNum(kTailHandleNum_);
    tilingData_.set_vTailHandleNum(vTailHandleNum_);
    tilingData_.set_keyStride0(keyStride0_);
    tilingData_.set_keyStride1(keyStride1_);
    tilingData_.set_keyStride2(keyStride2_);
    tilingData_.set_valueStride0(valueStride0_);
    tilingData_.set_valueStride1(valueStride1_);
    tilingData_.set_valueStride2(valueStride2_);
    tilingData_.set_kHeadSize(kHeadSize_);
    tilingData_.set_vHeadSize(vHeadSize_);
    tilingData_.set_batch(batch_);
    tilingData_.set_seqLen(seqLen_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_numBlocks(numBlocks_);
    tilingData_.set_blockSize(blockSize_);
    GenTilingKey();
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterPaKvCacheTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "totalCoreNum: " << totalCoreNum_ << std::endl;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "blockFactor: " << blockFactor_ << std::endl;
    info << "tailBlockFactor: " << tailBlockFactor_ << std::endl;
    info << "kHandleNumPerCore: " << kHandleNumPerCore_ << std::endl;
    if (inOutMode_ == DUAL_IN_OUT) {
        info << "vHandleNumPerCore: " << vHandleNumPerCore_ << std::endl;
    }
    info << "kLoopNum: " << kLoopNum_ << std::endl;
    if (inOutMode_ == DUAL_IN_OUT) {
        info << "vLoopNum: " << vLoopNum_ << std::endl;
    }
    info << "kHandleNumPerLoop: " << kHandleNumPerLoop_ << std::endl;
    if (inOutMode_ == DUAL_IN_OUT) {
        info << "vHandleNumPerLoop: " << vHandleNumPerLoop_ << std::endl;
    }
    info << "kTailHandleNum: " << kTailHandleNum_ << std::endl;
    if (inOutMode_ == DUAL_IN_OUT) {
        info << "vTailHandleNum: " << vTailHandleNum_ << std::endl;
    }
    info << "keyStride0: " << keyStride0_ << std::endl;
    info << "keyStride1: " << keyStride1_ << std::endl;
    info << "keyStride2: " << keyStride2_ << std::endl;
    if (inOutMode_ == DUAL_IN_OUT) {
        info << "valueStride0: " << valueStride0_ << std::endl;
        info << "valueStride1: " << valueStride1_ << std::endl;
        info << "valueStride2: " << valueStride2_ << std::endl;
    }
    info << "kHeadSize: " << kHeadSize_ << std::endl;
    if (inOutMode_ == DUAL_IN_OUT) {
        info << "vHeadSize: " << vHeadSize_ << std::endl;
    }
    info << "seqLen: " << seqLen_ << std::endl;
    info << "numBlocks: " << numBlocks_ << std::endl;
    info << "blockSize: " << blockSize_ << std::endl;
    info << "tilingKey: " << tilingKey_ << std::endl;
}

ge::graphStatus Tiling4ScatterPaKvCache(gert::TilingContext *context_)
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        ScatterPaKvCacheMembaseTiling tiling(context_);
        return tiling.DoTiling();
    }

    ScatterPaKvCacheTiling tiling(context_, DUAL_IN_OUT);
    return tiling.DoTiling();
}

ge::graphStatus Tiling4ScatterPaCache(gert::TilingContext *context_)
{
    ScatterPaKvCacheTiling tiling(context_, SINGLE_IN_OUT);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4ScatterPaKvCache(gert::TilingParseContext *context_)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ScatterPaKvCache op.
IMPL_OP_OPTILING(ScatterPaKvCache)
    .Tiling(Tiling4ScatterPaKvCache)
    .TilingParse<ScatterPaKvCacheCompileInfo>(TilingPrepare4ScatterPaKvCache);
IMPL_OP_OPTILING(ScatterPaCache)
    .Tiling(Tiling4ScatterPaCache)
    .TilingParse<ScatterPaKvCacheCompileInfo>(TilingPrepare4ScatterPaKvCache);

} // namespace optiling