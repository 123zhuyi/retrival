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
 * \file real_tiling.cpp
 * \brief real tiling cpp
 */

#include <cmath>
#include <type_traits>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/fp16.h"
#include "../op_kernel/real_tiling.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
using namespace Ops::Base;

constexpr static int32_t INDEX_INPUT_X = 0;
constexpr static int32_t INDEX_OUTPUT_Y = 0;
constexpr static size_t WORKSPACE_COUNT = 1;

static inline int64_t AlignCeil(int64_t value, int64_t align)
{
    if (align > 0) {
        return static_cast<int64_t>(((value + align - 1) / align) * align);
    }
    return value;
}

// GatherMask inplace requires: count * COMPLEX_COEFFICIENT * sizeof(T) % 256 == 0
// For complex: count * 2 * sizeof(T) % 256 == 0 → count * sizeof(T) % 128 == 0
// So use 128B alignment for complex, 32B for real.
constexpr int64_t GATHER_MASK_ALIGN_BYTES = 128;
constexpr int64_t COMPLEX_COEFF = 2;

static ge::graphStatus CheckDtypeIsValid(gert::TilingContext* context, ge::DataType input, ge::DataType output)
{
    std::set<ge::DataType> inputDtype = {ge::DT_COMPLEX32, ge::DT_COMPLEX64, ge::DT_FLOAT16, ge::DT_FLOAT};
    std::set<ge::DataType> outputDtype = {ge::DT_FLOAT16, ge::DT_FLOAT};

    OP_CHECK_IF(
        inputDtype.count(input) == 0,
        OP_LOGE(
            context->GetNodeName(), "Input dtype(%s) is invalid, it should be complex32, complex64, float16 or float.",
            Ops::Base::ToString(input).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputDtype.count(output) == 0,
        OP_LOGE(
            context->GetNodeName(), "Output dtype(%s) is invalid, it should be float16 or float.",
            Ops::Base::ToString(output).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetSingleCore(
    RealTilingParam& tilingParam, const int64_t& ubAvailable, const int64_t& alignSize, int64_t& ubPartDataNum)
{
    // 检查是否满足 inplace GatherMask 的 256B 对齐约束
    int64_t totalSourceBytes = tilingParam.totalLength * COMPLEX_COEFF * tilingParam.dataTypeLength;
    bool useNonInplace = tilingParam.isComplexInput && (totalSourceBytes < 256 || totalSourceBytes % 256 != 0);
    if (useNonInplace) {
        // 不满足 inplace 约束: 切换到 non-inplace (inQueue 2x + outQueue 1x) × 2缓冲 = 6x
        int64_t ubMultiplier = 6;
        int64_t ubPartLength = ubAvailable / ubMultiplier / REAL_BUFFER_NUM;
        int64_t ubPartBlockNum = ubPartLength;
        if (alignSize > 0) {
            ubPartBlockNum = ubPartLength / alignSize;
        }
        ubPartDataNum = (ubPartBlockNum * alignSize) / tilingParam.dataTypeLength;
    }
    tilingParam.totalUsedCoreNum = 1;
    tilingParam.tailBlockNum = 0;
    tilingParam.ubPartDataNum = ubPartDataNum;
    tilingParam.smallCoreDataNum = tilingParam.totalLength;
    tilingParam.smallCoreLoopNum = 1;
    tilingParam.smallCoreTailDataNum = tilingParam.totalLength;
    tilingParam.bigCoreDataNum = 0;
    tilingParam.bigCoreLoopNum = 0;
    tilingParam.bigCoreTailDataNum = 0;
    tilingParam.useNonInplace = useNonInplace ? 1 : 0;
}

static void SetBigCore(
    const int64_t& bigCoreDataNum, const int64_t& ubPartDataNum, int64_t& bigCoreLoopNum, int64_t& bigCoreTailDataNum)
{
    bigCoreLoopNum = Ops::Base::CeilDiv(bigCoreDataNum, ubPartDataNum);
    bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
    if (bigCoreTailDataNum == 0) {
        bigCoreTailDataNum = ubPartDataNum;
    }
}

static ge::graphStatus CalcRealTilingParam(
    RealTilingParam& tilingParam, ge::DataType inputDtype, ge::DataType outputDtype)
{
    int32_t outputSize = ge::GetSizeByDataType(outputDtype);
    tilingParam.dataTypeLength = static_cast<int64_t>(outputSize);
    tilingParam.isComplexInput = (inputDtype == ge::DT_COMPLEX32 || inputDtype == ge::DT_COMPLEX64);
    int64_t alignSize = tilingParam.isComplexInput ? GATHER_MASK_ALIGN_BYTES : REAL_BLOCK_SIZE;
    int64_t ubAvailable = std::max(static_cast<int64_t>(1), (tilingParam.totalUbSize - RESERVED_UB_SIZE));
    int64_t ubMultiplier = tilingParam.isComplexInput ? 4 : 2;
    int64_t ubPartLength = ubAvailable / ubMultiplier / REAL_BUFFER_NUM;
    int64_t ubPartBlockNum = ubPartLength / alignSize;
    int64_t ubPartDataNum = (ubPartBlockNum * alignSize) / tilingParam.dataTypeLength;
    int64_t totalBytes = tilingParam.totalLength * tilingParam.dataTypeLength;
    int64_t totalBytesAlign = AlignCeil(totalBytes, alignSize);
    int64_t totalBlocks = totalBytesAlign / alignSize;
    int64_t coreNum = 1;
    if (ubPartDataNum < tilingParam.totalLength) {
        int64_t maxCoreNum = totalBlocks;
        coreNum = std::min(tilingParam.totalCoreNum, maxCoreNum);
    }
    if (coreNum == 1) {
        SetSingleCore(tilingParam, ubAvailable, alignSize, ubPartDataNum);
        return ge::GRAPH_SUCCESS;
    }

    int64_t everyCoreBlockNum = totalBlocks / coreNum;
    int64_t tailBlockNum = totalBlocks % coreNum;

    int64_t smallCoreDataNum = everyCoreBlockNum * alignSize / tilingParam.dataTypeLength;
    int64_t smallCoreLoopNum = Ops::Base::CeilDiv(smallCoreDataNum, ubPartDataNum);
    int64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    if (smallCoreTailDataNum == 0) {
        smallCoreTailDataNum = ubPartDataNum;
    }

    int64_t bigCoreDataNum = 0;
    int64_t bigCoreLoopNum = 0;
    int64_t bigCoreTailDataNum = 0;
    if (tailBlockNum > 0) {
        int64_t bigCoreBlockNum = everyCoreBlockNum + 1;
        bigCoreDataNum = bigCoreBlockNum * alignSize / tilingParam.dataTypeLength;
        SetBigCore(bigCoreDataNum, ubPartDataNum, bigCoreLoopNum, bigCoreTailDataNum);
    }
    tilingParam.totalUsedCoreNum = coreNum;
    tilingParam.tailBlockNum = tailBlockNum;
    tilingParam.ubPartDataNum = ubPartDataNum;
    tilingParam.smallCoreDataNum = smallCoreDataNum;
    tilingParam.smallCoreLoopNum = smallCoreLoopNum;
    tilingParam.smallCoreTailDataNum = smallCoreTailDataNum;
    tilingParam.bigCoreDataNum = bigCoreDataNum;
    tilingParam.bigCoreLoopNum = bigCoreLoopNum;
    tilingParam.bigCoreTailDataNum = bigCoreTailDataNum;

    return ge::GRAPH_SUCCESS;
}

class RealMemBaseTilingClass : public TilingBaseClass {
public:
    explicit RealMemBaseTilingClass(gert::TilingContext* context) : TilingBaseClass(context) {}

    void Reset(gert::TilingContext* context) override { TilingBaseClass::Reset(context); }

protected:
    ge::graphStatus GetPlatformInfo() override
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
            auto compileInfoPtr = reinterpret_cast<const RealCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
            tilingParam_.totalCoreNum = compileInfoPtr->totalCoreNum;
            tilingParam_.totalUbSize = static_cast<int64_t>(compileInfoPtr->ubSizePlatForm);
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override { return ge::GRAPH_SUCCESS; }

    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    bool IsCapable() override { return (!Ops::Base::IsRegbaseSocVersion(context_)); }

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus GetShapeAttrsInfo() override { return ge::GRAPH_SUCCESS; }

    uint64_t GetTilingKey() const override { return context_->GetTilingKey(); }

    NpuArch npuArch_;
    RealTilingParam tilingParam_;
};

static ge::graphStatus SetTilingTilingKeyForReal(
    gert::TilingContext* context, int64_t& tilingKey, const ge::DataType inputDtype)
{
    // Set tilingKey based on input dtype to distinguish between complex extraction and identity
    switch (inputDtype) {
        case ge::DT_COMPLEX32:
            tilingKey = static_cast<int64_t>(RealTilingKey::TILINGKEY_COMPLEX32);
            break;
        case ge::DT_COMPLEX64:
            tilingKey = static_cast<int64_t>(RealTilingKey::TILINGKEY_COMPLEX64);
            break;
        case ge::DT_FLOAT16:
            tilingKey = static_cast<int64_t>(RealTilingKey::TILINGKEY_FLOAT16);
            break;
        case ge::DT_FLOAT:
            tilingKey = static_cast<int64_t>(RealTilingKey::TILINGKEY_FLOAT);
            break;
        default:
            OP_LOGE(
                context->GetNodeName(), "set tilingKey fail: unsupported input dtype %d", static_cast<int>(inputDtype));
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RealMemBaseTilingClass::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling4Real enter.");

    auto xShape = context_->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);

    int64_t totalLength = xShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        (totalLength == 0), OP_LOGE(context_->GetNodeName(), "Real input shape size is 0"), return ge::GRAPH_FAILED);

    auto inputDesc = context_->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    ge::DataType outputDtype = context_->GetOutputDesc(INDEX_OUTPUT_Y)->GetDataType();

    auto ret = CheckDtypeIsValid(context_, inputDtype, outputDtype);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    tilingParam_.totalLength = totalLength;

    // Calculate tiling parameters
    OP_CHECK_IF(
        CalcRealTilingParam(tilingParam_, inputDtype, outputDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CalcRealTilingParam fail."), return ge::GRAPH_FAILED);

    // Set tiling key
    int64_t tilingKey = 0;
    OP_CHECK_IF(
        SetTilingTilingKeyForReal(context_, tilingKey, inputDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "set TilingKey fail."), return ge::GRAPH_FAILED);
    tilingParam_.tilingKey = tilingKey;

    OP_LOGD(context_->GetNodeName(), "Tiling4Real exit.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RealMemBaseTilingClass::PostTiling()
{
    RealTilingData* tiling = context_->GetTilingData<RealTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);

    tiling->totalUsedCoreNum = tilingParam_.totalUsedCoreNum;
    tiling->tailBlockNum = tilingParam_.tailBlockNum;
    tiling->ubPartDataNum = tilingParam_.ubPartDataNum;
    tiling->smallCoreDataNum = tilingParam_.smallCoreDataNum;
    tiling->smallCoreLoopNum = tilingParam_.smallCoreLoopNum;
    tiling->smallCoreTailDataNum = tilingParam_.smallCoreTailDataNum;
    tiling->bigCoreDataNum = tilingParam_.bigCoreDataNum;
    tiling->bigCoreLoopNum = tilingParam_.bigCoreLoopNum;
    tiling->bigCoreTailDataNum = tilingParam_.bigCoreTailDataNum;
    tiling->tilingKey = tilingParam_.tilingKey;
    tiling->useNonInplace = tilingParam_.useNonInplace;

    // 设置 userWorkspace
    size_t* userWorkspaceSize = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;

    context_->SetBlockDim(tiling->totalUsedCoreNum);
    context_->SetTilingKey(tiling->tilingKey);

    OP_LOGD(
        context_->GetNodeName(),
        "tilingData: totalUsedCoreNum=%ld, tailBlockNum=%ld, ubPartDataNum=%ld, "
        "smallCoreDataNum=%ld, smallCoreLoopNum=%ld, smallCoreTailDataNum=%ld, "
        "bigCoreDataNum=%ld, bigCoreLoopNum=%ld, bigCoreTailDataNum=%ld, tilingKey=%ld, "
        "useNonInplace=%ld",
        tiling->totalUsedCoreNum, tiling->tailBlockNum, tiling->ubPartDataNum, tiling->smallCoreDataNum,
        tiling->smallCoreLoopNum, tiling->smallCoreTailDataNum, tiling->bigCoreDataNum, tiling->bigCoreLoopNum,
        tiling->bigCoreTailDataNum, tiling->tilingKey, tiling->useNonInplace);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Real(gert::TilingContext* context)
{
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4Real(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4Real enter.");

    auto compileInfo = context->GetCompiledInfo<RealCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "TilingPrepare4Real fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "TilingPrepare4Real fail to get ub size."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Real, RealMemBaseTilingClass, 1000);

IMPL_OP_OPTILING(Real).Tiling(Tiling4Real).TilingParse<RealCompileInfo>(TilingPrepare4Real);
} // namespace optiling
