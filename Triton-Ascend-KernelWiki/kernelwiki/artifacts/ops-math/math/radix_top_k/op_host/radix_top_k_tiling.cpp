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
 * \file radix_top_k_tiling.cpp
 * \brief
 */

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "radix_top_k_tiling.h"
#include "../op_kernel/radix_top_k_struct.h"
#include "../op_kernel/radix_top_k_tiling_key.h"

namespace optiling {
namespace RadixTopK {
static constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t CACHE_LINE = 512;
static constexpr uint64_t MAX_TILE_LEN = 12032;
static constexpr uint64_t LARGE_SORT_LEN = 1440000000;
static constexpr uint64_t LARGE_TILE_LEN = 10496;
static constexpr uint64_t MAX_TILE_NUM_IN_UB = 6144;
static constexpr uint64_t INPUT_X = 0;
static constexpr uint64_t NUM_VALUE_2BIT = 4;
static constexpr uint64_t BIT_PER_BYTE = 8;
static constexpr uint64_t BUFFER_NUM_IN_OUT = 4;
static constexpr uint64_t ATTR_INDEX_SORTED = 0;
static constexpr uint64_t ATTR_INDEX_LARGEST = 2;
static constexpr uint64_t WS_INT32_PER_TILE = 5;
// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus RadixTopKTiling::GetPlatformInfo()
{
    OP_LOGD(opName_, "RadixTopKTiling GetPlatformInfo.");
    auto compileInfo = static_cast<const RadixTopKCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<uint64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus RadixTopKTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "RadixTopKTiling GetShapeAttrsInfo.");
    // 获取输入shape和dtype
    const gert::StorageShape* xShape = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    uint64_t batch = 1;
    uint64_t dimNum = xShape->GetShape().GetDimNum();
    for (uint64_t i = 0; i < dimNum - 1; i++) {
        batch *= xShape->GetShape().GetDim(i);
    }
    tilingData_.batch = batch;
    tilingData_.sortLen = xShape->GetShape().GetDim(dimNum - 1);

    auto tensorK = context_->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorK);
    const int32_t* constDataPtr = tensorK->GetData<int32_t>();
    OP_CHECK_IF(constDataPtr == nullptr, OP_LOGE(opName_, "Get const data k failed."), return ge::GRAPH_FAILED);
    int32_t kValue = static_cast<int32_t>(*constDataPtr);
    OP_CHECK_IF(kValue == 0, OP_LOGE(opName_, "Get k equals to zero."), return ge::GRAPH_FAILED);
    tilingData_.kValue = kValue;

    auto attrs = context_->GetAttrs();
    tilingData_.sorted = *attrs->GetAttrPointer<bool>(ATTR_INDEX_SORTED);
    tilingData_.largest = *attrs->GetAttrPointer<bool>(ATTR_INDEX_LARGEST);
    isLargeShape_ = tilingData_.sortLen > LARGE_SORT_LEN;

    xDType_ = context_->GetInputDesc(INPUT_X)->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDType_);

    return ge::GRAPH_SUCCESS;
}

// 3、计算数据切分TilingData
ge::graphStatus RadixTopKTiling::DoOpTiling()
{
    OP_LOGD(opName_, "RadixTopKTiling DoOpTiling.");
    bool isCalcSuccess = false;
    if (!isLargeShape_) {
        isCalcSuccess = CalcTilingParams(tilingData_.sortLen);
        if (!isCalcSuccess) {
            isLargeShape_ = true;
            isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen);
        }
    } else {
        isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen);
    }
    OP_CHECK_IF((!isCalcSuccess), OP_LOGE(opName_, "RadixTopK Tiling failed."), return ge::GRAPH_FAILED);

    blockDim_ = tilingData_.formerCoreNum + tilingData_.tailCoreNum;
    tilingData_.coreNum = blockDim_;
    // 仅 UB 变体可复用 indices 内存（WS 变体的 tileTopK/tileHist 与 CopyOutResult 跨 core 冲突）
    if (isLargeShape_) {
        tilingData_.needWorkspace = true;
    } else {
        uint64_t wsInt32Cnt = tilingData_.coreNum * (NUM_VALUE_2BIT + 2);
        tilingData_.needWorkspace = (tilingData_.kValue <= wsInt32Cnt);
    }
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus RadixTopKTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t RadixTopKTiling::GetTilingKey() const
{
    OP_LOGD(opName_, "RadixTopKTiling GetTilingKey.");

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tilingData_.sorted, tilingData_.largest, isLargeShape_);
    OP_LOGD(opName_, "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(opName_, "sorted, largest, isLargeShape is: [%d, %d, %d]", tilingData_.sorted, tilingData_.largest, isLargeShape_);

    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus RadixTopKTiling::GetWorkspaceSize()
{
    if (!tilingData_.needWorkspace) {
        workspaceSize_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    if (isLargeShape_) {
        workspaceSize_ = (NUM_VALUE_2BIT * tilingData_.coreNum + tilingData_.coreNum +
                          tilingData_.coreNum + WS_INT32_PER_TILE * tilingData_.totalTileNum) * sizeof(int32_t);
    } else {
        workspaceSize_ = (NUM_VALUE_2BIT * tilingData_.coreNum + tilingData_.coreNum + tilingData_.coreNum) * sizeof(int32_t);
    }
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus RadixTopKTiling::PostTiling()
{
    OP_LOGD(opName_, "RadixTopKTiling PostTiling.");

    // 设置workspace大小
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    // 用到SyncAll，需要设置为batch mode模式
    context_->SetScheduleMode(1);

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(RadixTopKTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(RadixTopKTilingData));

    return ge::GRAPH_SUCCESS;
}

bool RadixTopKTiling::IsCapable()
{
    return true;
}

void RadixTopKTiling::PrintTilingData()
{
    OP_LOGD(opName_, "formerCoreNum:    %lu.", tilingData_.formerCoreNum);
    OP_LOGD(opName_, "tailCoreNum:      %lu.", tilingData_.tailCoreNum);
    OP_LOGD(opName_, "totalTileNum:     %lu.", tilingData_.totalTileNum);
    OP_LOGD(opName_, "formerTileNum:    %lu.", tilingData_.formerTileNum);
    OP_LOGD(opName_, "tailTileNum:      %lu.", tilingData_.tailTileNum);
    OP_LOGD(opName_, "formerTileLen:    %lu.", tilingData_.formerTileLen);
    OP_LOGD(opName_, "tailTileLen:      %lu.", tilingData_.tailTileLen);
    OP_LOGD(opName_, "batch:            %lu.", tilingData_.batch);
    OP_LOGD(opName_, "sortLen:          %lu.", tilingData_.sortLen);
    OP_LOGD(opName_, "kValue:           %lu.", tilingData_.kValue);
    OP_LOGD(opName_, "largest:          %d.", tilingData_.largest);
    OP_LOGD(opName_, "sorted:           %d.", tilingData_.sorted);
    OP_LOGD(opName_, "needWorkspace:    %d.", tilingData_.needWorkspace);
}

void RadixTopKTiling::CalcTileDistribution(const uint64_t &dataNum, const uint64_t &formerTileLen)
{
    uint64_t totalTileNum = Ops::Base::CeilDiv(dataNum, formerTileLen);
    uint64_t formerTileNum = Ops::Base::CeilDiv(totalTileNum, totalCoreNum_);
    uint64_t tailTileNum = totalTileNum / totalCoreNum_;
    uint64_t formerCoreNum = totalTileNum % totalCoreNum_;
    uint64_t tailCoreNum = tailTileNum ? (totalCoreNum_ - formerCoreNum) : 0;
    uint64_t totalProcessedTiles = formerTileNum * formerCoreNum + tailTileNum * tailCoreNum;
    uint64_t tailTileLen = dataNum - formerTileLen * (totalProcessedTiles - 1);

    tilingData_.formerCoreNum = formerCoreNum;
    tilingData_.tailCoreNum = tailCoreNum;
    tilingData_.totalTileNum = totalTileNum;
    tilingData_.formerTileNum = formerTileNum;
    tilingData_.tailTileNum = tailTileNum;
    tilingData_.formerTileLen = formerTileLen;
    tilingData_.tailTileLen = tailTileLen;
}

template<typename CheckUbFn>
bool RadixTopKTiling::TryCalcTileDistribution(
    const uint64_t &dataNum, uint64_t startTileLen,
    uint64_t minTileLen, uint64_t step, CheckUbFn &&checkUb)
{
    for (uint64_t tileLen = startTileLen; tileLen >= minTileLen; tileLen -= step) {
        if (checkUb(tileLen)) {
            CalcTileDistribution(dataNum, tileLen);
            return true;
        }
    }
    OP_LOGD(opName_, "Cannot find suitable tileLen to do RadixTopK.");
    return false;
}

bool RadixTopKTiling::CalcLargeTilingParams(const uint64_t &dataNum)
{
    uint64_t dataAlign = CACHE_LINE / xDtypeSize_;
    uint64_t minTileLen = dataAlign;

    auto checkUb = [&](uint64_t tileLen) -> bool {
        uint64_t cmpMaskSize = tileLen / 8;
        uint64_t ubUsed = 16 * tileLen + cmpMaskSize + MAX_TILE_NUM_IN_UB * sizeof(int32_t) + BLOCK_SIZE;
        return ubUsed <= ubSize_;
    };

    return TryCalcTileDistribution(dataNum, LARGE_TILE_LEN, minTileLen, dataAlign, checkUb);
}

bool RadixTopKTiling::CalcTilingParams(const uint64_t &dataNum)
{
    uint64_t coreNum = totalCoreNum_;
    uint64_t dataAlign = CACHE_LINE / xDtypeSize_;
    uint64_t minTileLen = dataAlign;

    auto checkUb = [&](uint64_t tileLen) -> bool {
        if (tileLen == 0) return false;
        uint64_t totalTn = (dataNum + tileLen - 1) / tileLen;
        if (totalTn == 0) totalTn = 1;
        uint64_t tn = (totalTn + coreNum - 1) / coreNum;
        uint64_t tnAlign = Ops::Base::CeilAlign(tn, static_cast<uint64_t>(8));
        uint64_t tileHistSize = NUM_VALUE_2BIT * tnAlign * sizeof(int32_t);
        uint64_t cmpMaskSize = tileLen / 8;
        uint64_t tempBuf = (tileHistSize > cmpMaskSize) ? tileHistSize : cmpMaskSize;
        uint64_t ubUsed = BUFFER_NUM_IN_OUT * sizeof(int32_t) * tileLen + NUM_VALUE_2BIT * tnAlign + tempBuf + BLOCK_SIZE;
        return ubUsed <= ubSize_;
    };

    uint64_t startTileLen = MAX_TILE_LEN;
    uint64_t utilCap = Ops::Base::CeilAlign(Ops::Base::CeilDiv(dataNum, coreNum), dataAlign);
    if (utilCap < startTileLen) {
        startTileLen = utilCap;
    }

    return TryCalcTileDistribution(dataNum, startTileLen, minTileLen, dataAlign, checkUb);
}

void RadixTopKTiling::SetTilingData()
{
    OP_LOGD(opName_, "RadixTopKTiling SetTilingData.");
    RadixTopKTilingData* tilingData =
        context_->GetTilingData<RadixTopKTilingData>();
    tilingData->formerCoreNum = tilingData_.formerCoreNum;
    tilingData->tailCoreNum = tilingData_.tailCoreNum;
    tilingData->totalTileNum = tilingData_.totalTileNum;
    tilingData->formerTileNum = tilingData_.formerTileNum;
    tilingData->tailTileNum = tilingData_.tailTileNum;
    tilingData->formerTileLen = tilingData_.formerTileLen;
    tilingData->tailTileLen = tilingData_.tailTileLen;
    tilingData->coreNum = tilingData_.coreNum;
    tilingData->batch = tilingData_.batch;
    tilingData->sortLen = tilingData_.sortLen;
    tilingData->kValue = tilingData_.kValue;
    tilingData->sorted = tilingData_.sorted;
    tilingData->largest = tilingData_.largest;
    tilingData->needWorkspace = tilingData_.needWorkspace;
}
} // namespace RadixTopK

static ge::graphStatus TilingRadixTopK(gert::TilingContext* context)
{
    OP_LOGD(context, "RadixTopKTiling start.");
    RadixTopK::RadixTopKTiling tilingOp(context);
    auto ret = tilingOp.DoTiling();
    OP_CHECK_IF(
        (ret == ge::GRAPH_FAILED), OP_LOGD(context, "RadixTopKTiling tiling failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "RadixTopKTiling end.");
        
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForRadixTopK(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For RadixTopK start.");
    auto compileInfo = context->GetCompiledInfo<RadixTopKCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        OP_LOGE(context, "coreNum %d", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For RadixTopK end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RadixTopK)
    .Tiling(TilingRadixTopK)
    .TilingParse<RadixTopKCompileInfo>(TilingPrepareForRadixTopK)
    .TilingInputsDataDependency({1});
} // namespace optiling
