/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_grouped_mat_mul_tiling.cc
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "context_util.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../allto_allv_grouped_mat_mul_tiling_base.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_quant_tiling.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
bool AlltoAllvGmmQuantTiling::IsCapable()
{
    // only support hifloat8
    if (gmmXDataType_ != ge::DT_HIFLOAT8) {
        return false;
    }
    if (gmmWeightDataType_ != ge::DT_HIFLOAT8) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvGmmQuantTiling is capable.");
    return true;
}

ge::graphStatus AlltoAllvGmmQuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "start quant GetPlatformInfo.");
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end quant GetPlatformInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeAttrsInfo.");
    if (GetCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoOpTiling.");
    if (CheckGmmDType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmDType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivCoreNum_, aicCoreNum_, aivCoreNum_));
    context_->SetTilingKey(GetTilingKey());
    OP_TILING_CHECK(SetHcclTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end DoOpTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoLibApiTiling.");
    uint64_t maxMSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < e_; expertIdx++) {
        mSize = 0;
        for (uint64_t rankIdx = 0; rankIdx < epWorldSize_; rankIdx++) {
            mSize += recvCounts[rankIdx * e_ + expertIdx];
        }
        maxMSize = std::max(mSize, maxMSize);
    }
    if (maxMSize != 0) {
        auto &gmmQuantTilingData = tilingData->gmmQuantTilingData;
        SetGMMQuantParams(gmmQuantTilingData);
        SetTilingArray(gmmQuantTilingData, maxMSize, n1_, h1_);
        SetTilingParams(gmmQuantTilingData, maxMSize, n1_, h1_);
        PrintGMMQuantTilingData(gmmQuantTilingData);
    }
    if (bs_ != 0) {
        auto &mmQuantTilingData = tilingData->mmQuantTilingData;
        SetGMMQuantParams(mmQuantTilingData);
        SetTilingArray(mmQuantTilingData, bs_, n2_, h2_);
        SetTilingParams(mmQuantTilingData, bs_, n2_, h2_);
        PrintGMMQuantTilingData(mmQuantTilingData);
    }
    OP_LOGD(context_->GetNodeName(), "end DoLibApiTiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmQuantTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(ADD_TPL_HIF8, hasSharedExpertFlag_, transGmmWeight_, transMmWeight_);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmQuantTiling::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "start GetWorkspaceSize.");
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "can not get workspace."),
        return ge::GRAPH_FAILED);
    
    uint64_t permuteOutSize = permuteOutFlag_ ? 0 : (a_ * h1_ * GetSizeByDataType(gmmXDataType_));
    // 将 permuteOutSize 对齐到 512 字节
    const uint64_t tensorListSize = 512;
    if (permuteOutSize % tensorListSize != 0) {
        permuteOutSize = (permuteOutSize + tensorListSize - 1) & ~(tensorListSize - 1);
    }
    uint64_t groupListSize = sizeof(int64_t) * e_; // GMM计算所需的groupList GM空间大小
    // tensorListSize为kernel侧tensorlist开辟的空间
    workspaces[0] = libApiWorkSpaceSize_ + permuteOutSize + groupListSize + tensorListSize;
    OP_LOGD(context_->GetNodeName(), "end GetWorkspaceSize.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "start PostTiling.");
    tilingData->taskTilingInfo.BSK = bsk_;
    tilingData->taskTilingInfo.BS = bs_;
    tilingData->taskTilingInfo.H1 = h1_;
    tilingData->taskTilingInfo.H2 = h2_;
    tilingData->taskTilingInfo.A = a_;
    tilingData->taskTilingInfo.N1 = n1_;
    tilingData->taskTilingInfo.N2 = n2_;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize_;
    tilingData->taskTilingInfo.e = e_;
    tilingData->taskTilingInfo.mainLoopExpertNum = e_;
    tilingData->taskTilingInfo.tailLoopExpertNum = 0;
    tilingData->taskTilingInfo.totalLoopCount = e_;
    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCounts[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCounts[i];
    }
    PrintTaskTilingInfo(tilingData->taskTilingInfo);
    OP_LOGD(context_->GetNodeName(), "end PostTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvGmmQuantTiling::SetGMMQuantParams(
    Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const
{
    gmmQuantTilingData.gmmQuantParams.groupNum = SINGLE_GROUP_NUM;
    gmmQuantTilingData.gmmQuantParams.activeType = GMM_ACT_TYPE_NONE;
    gmmQuantTilingData.gmmQuantParams.aQuantMode = PERTENSOR_QUANT_MODE;
    gmmQuantTilingData.gmmQuantParams.bQuantMode = PERTENSOR_QUANT_MODE;
    gmmQuantTilingData.gmmQuantParams.singleX = 0;
    gmmQuantTilingData.gmmQuantParams.singleW = 0;
    gmmQuantTilingData.gmmQuantParams.singleY = 0;
    gmmQuantTilingData.gmmQuantParams.groupType = 0;
    gmmQuantTilingData.gmmQuantParams.groupListType = 1;
    gmmQuantTilingData.gmmQuantParams.hasBias = 0;
    gmmQuantTilingData.gmmQuantParams.reserved = 0;
}

void AlltoAllvGmmQuantTiling::SetTilingArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const
{
    gmmQuantTilingData.gmmArray.mList[0] = static_cast<int32_t>(M);
    gmmQuantTilingData.gmmArray.kList[0] = static_cast<int32_t>(K);
    gmmQuantTilingData.gmmArray.nList[0] = static_cast<int32_t>(N);
}

void AlltoAllvGmmQuantTiling::SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const
{
    auto &mm = gmmQuantTilingData.mmTilingData;

    mm.M = M;
    mm.N = N;
    mm.Ka = K;
    mm.Kb = K;
    mm.usedCoreNum = aicCoreNum_;
    mm.isBias = 0;
    mm.dbL0A = DOUBLE_BUFFER;
    mm.dbL0B = DOUBLE_BUFFER;

    mm.baseM = std::min(static_cast<int32_t>(M), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseM = Ops::Base::CeilAlign(mm.baseM, static_cast<int32_t>(CUBE_BLOCK));
    mm.baseN = std::min(static_cast<int32_t>(N), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseN = Ops::Base::CeilAlign(mm.baseN, static_cast<int32_t>(CUBE_BLOCK));
    mm.baseK = std::min(static_cast<int32_t>(K), static_cast<int32_t>(BASIC_BLOCK_SIZE_128));
    mm.baseK = Ops::Base::CeilAlign(mm.baseK, static_cast<int32_t>(CUBE_REDUCE_BLOCK));

    mm.singleCoreM = std::min(static_cast<int32_t>(M), mm.baseM);
    mm.singleCoreN = std::min(static_cast<int32_t>(N), mm.baseN);
    mm.singleCoreK = K;

    uint64_t l0cRequired = static_cast<uint64_t>(mm.baseM) * mm.baseN * DATA_SIZE_L0C * DB_SIZE;
    mm.dbL0C = (l0cRequired <= l0cSize_) ? DB_SIZE : 1;

    mm.iterateOrder = 0U;

    uint64_t baseASize = static_cast<uint64_t>(mm.baseM) * mm.baseK;
    uint64_t baseBSize = static_cast<uint64_t>(mm.baseN) * mm.baseK;
    uint64_t baseL1Size = baseASize + baseBSize;

    OP_TILING_CHECK(baseL1Size == 0, OP_LOGW(context_->GetNodeName(), "baseL1Size cannot be zero."), return );

    uint64_t leftL1Size = l1Size_;

    uint64_t depthInit = leftL1Size / baseL1Size;
    depthInit = std::max(depthInit, static_cast<uint64_t>(1));

    uint64_t depthScale = depthInit;
    while (depthScale * mm.baseK % BASIC_BLOCK_SIZE_512 != 0 && depthScale > 1) {
        depthScale--;
    }
    depthScale = std::max(depthScale, static_cast<uint64_t>(1));

    mm.depthA1 = depthScale;
    mm.depthB1 = depthScale;

    mm.stepKa = (mm.depthA1 > 1) ? (mm.depthA1 / DB_SIZE) : 1;
    mm.stepKb = (mm.depthB1 > 1) ? (mm.depthB1 / DB_SIZE) : 1;

    OP_TILING_CHECK(mm.baseK == 0, OP_LOGW(context_->GetNodeName(), "baseK cannot be zero."), return );

    if (mm.stepKa * mm.baseK > mm.Ka) {
        mm.stepKa = Ops::Base::CeilDiv(mm.Ka, mm.baseK);
    }
    if (mm.stepKb * mm.baseK > mm.Kb) {
        mm.stepKb = Ops::Base::CeilDiv(mm.Kb, mm.baseK);
    }

    mm.depthA1 = mm.stepKa * DB_SIZE;
    mm.depthB1 = mm.stepKb * DB_SIZE;

    mm.stepM = 1;
    mm.stepN = 1;
}

void AlltoAllvGmmQuantTiling::PrintGMMQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;

    std::stringstream ss;
    ss << "MM Tiling: M=" << mm.M << ", N=" << mm.N << ", K=" << mm.Ka << ", usedCoreNum=" << mm.usedCoreNum <<
        ", baseM=" << mm.baseM << ", baseN=" << mm.baseN << ", baseK=" << mm.baseK << ", singleCoreM=" <<
        mm.singleCoreM << ", singleCoreN=" << mm.singleCoreN << ", singleCoreK=" << mm.singleCoreK << ", dbL0C=" <<
        mm.dbL0C << ", depthA1=" << mm.depthA1 << ", depthB1=" << mm.depthB1 << ", stepKa=" << mm.stepKa <<
        ", stepKb=" << mm.stepKb << ", stepM=" << mm.stepM << ", stepN=" << mm.stepN << ", iterateOrder=" <<
        mm.iterateOrder;

    ss << "\nQuant Params: groupNum=" << quantParams.groupNum << ", activeType=" << quantParams.activeType <<
        ", aQuantMode=" << quantParams.aQuantMode << ", bQuantMode=" << quantParams.bQuantMode << ", singleX=" <<
        quantParams.singleX << ", singleW=" << quantParams.singleW << ", singleY=" << quantParams.singleY <<
        ", groupType=" << quantParams.groupType << ", groupListType=" << quantParams.groupListType << ", hasBias=" <<
        quantParams.hasBias << ", reserved=" << quantParams.reserved;

    ss << "\nArray: mList[0]=" << gmmArray.mList[0] << ", kList[0]=" << gmmArray.kList[0] << ", nList[0]=" <<
        gmmArray.nList[0];

    OP_LOGI(context_->GetNodeName(), "AlltoAllvGmmQuantTiling TilingParams:\n%s", ss.str().c_str());
}

void AlltoAllvGmmQuantTiling::PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo) const
{
    std::stringstream ss;
    ss << "TaskTilingInfo: ";
    ss << "BSK=" << taskTilingInfo.BSK << ", BS=" << taskTilingInfo.BS << ", H1=" << taskTilingInfo.H1 << ", H2=" <<
        taskTilingInfo.H2 << ", A=" << taskTilingInfo.A << ", N1=" << taskTilingInfo.N1 << ", N2=" << taskTilingInfo.N2;
    ss << ", epWorldSize=" << taskTilingInfo.epWorldSize << ", e=" << taskTilingInfo.e;
    ss << ", mainLoopExpertNum=" << taskTilingInfo.mainLoopExpertNum << ", tailLoopExpertNum=" <<
        taskTilingInfo.tailLoopExpertNum << ", totalLoopCount=" << taskTilingInfo.totalLoopCount;
    ss << "\nSendCounts: ";
    for (int i = 0; i < e_ * epWorldSize_; i++) {
        if (taskTilingInfo.sendCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.sendCnt[i];
        }
    }
    ss << "\nRecvCounts: ";
    for (int i = 0; i < e_ * epWorldSize_; i++) {
        if (taskTilingInfo.recvCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.recvCnt[i];
        }
    }
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());
}

ge::graphStatus AlltoAllvGmmQuantTiling::CheckGmmDType() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmDType.");
    OP_TILING_CHECK((context_->GetInputDesc(GMM_X_INDEX) == nullptr) ||
        (context_->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    auto gmmXDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmXDataType != ge::DT_HIFLOAT8,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmX only support hifloat8."), return ge::GRAPH_FAILED);
    auto gmmWeightDataType = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmWeightDataType != ge::DT_HIFLOAT8,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmWeight only support hifloat8."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GMM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmXScale returned null."), return ge::GRAPH_FAILED);
    auto gmmXScaleDataType = context_->GetOptionalInputDesc(GMM_X_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmXScaleDataType != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmXScale only support float32."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmWeightScale returned null."), return ge::GRAPH_FAILED);
    auto gmmWeightScaleDataType = context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmWeightScaleDataType != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmWeightScale only support float32."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOutputDesc y returned null."), return ge::GRAPH_FAILED);
    auto gmmYDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(gmmYDataType != ge::DT_FLOAT16 && gmmYDataType != ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmY only support float16 and bfloat16."),
        return ge::GRAPH_FAILED);
    if (permuteOutFlag_) {
        // check permuteOut dtype
        tilingData->isPermuteOut = true;
        OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX) == nullptr,
            OP_LOGE(context_->GetNodeName(), "GetOutputDesc permuteOut returned null."), return ge::GRAPH_FAILED);
        auto permuteOutDataType = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(permuteOutDataType != gmmXDataType,
            OP_LOGE(context_->GetNodeName(), "Unsupported dataType, permuteOut only support hifloat8."),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckGmmDType.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::CheckMmDType() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmDType.");
    if (!hasSharedExpertFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((context_->GetOptionalInputDesc(MM_X_INDEX) == nullptr) ||
        (context_->GetOptionalInputDesc(MM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(context_->GetNodeName(), "GetOptionalInputDesc mmX or mmWeight returned null."),
        return ge::GRAPH_FAILED);
    auto mmXDataType = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(mmXDataType != ge::DT_HIFLOAT8,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmX only support hifloat8."), return ge::GRAPH_FAILED);
    auto mmWeightDataType = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(mmWeightDataType != ge::DT_HIFLOAT8,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmWeight only support hifloat8."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOptionalInputDesc mmXScale returned null."), return ge::GRAPH_FAILED);
    auto mmXScaleDataType = context_->GetOptionalInputDesc(MM_X_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(mmXScaleDataType != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmXScale only support float32."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOptionalInputDesc mmWeightScale returned null."), return ge::GRAPH_FAILED);
    auto mmWeightScaleDataType = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(mmWeightScaleDataType != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmWeightScale only support float32."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_MM_Y_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOutputDesc y returned null."), return ge::GRAPH_FAILED);
    auto mmYDataType = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(mmYDataType != ge::DT_FLOAT16 && mmYDataType != ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmY only support float16 and bfloat16."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckMmDType.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::CheckQuantMode() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantMode.");
    // gmmXQuantMode
    OP_TILING_CHECK(gmmXQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmXQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    auto gmmXQuantMode = *gmmXQuantModePtr_;
    OP_TILING_CHECK(gmmXQuantMode != PERTENSOR_QUANT_MODE,
        OP_LOGE(context_->GetNodeName(), "gmmXQuantMode only support 1(pertensor mode)."), return ge::GRAPH_FAILED);
    // gmmWeightQuantMode
    OP_TILING_CHECK(gmmWeightQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmWeightQuantMode attr can not be null."), return ge::GRAPH_FAILED);
    auto gmmWeightQuantMode = *gmmWeightQuantModePtr_;
    OP_TILING_CHECK(gmmWeightQuantMode != PERTENSOR_QUANT_MODE,
        OP_LOGE(context_->GetNodeName(), "gmmWeightQuantMode only support 1(pertensor mode)."),
        return ge::GRAPH_FAILED);
    // check gmmXScale shape
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_X_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmXScale input shape can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum() != DIM_ONE ||
        context_->GetOptionalInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO) != DIM_ONE,
        OP_LOGE(context_->GetNodeName(), "gmmXScale input shape should be [1]"), return ge::GRAPH_FAILED);
    // check gmmWeightScale shape
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmWeightScale input shape can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum() != DIM_ONE ||
        context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO) != DIM_ONE,
        OP_LOGE(context_->GetNodeName(), "gmmWeightScale input shape should be [1]."), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // mmXQuantMode
        OP_TILING_CHECK(mmXQuantModePtr_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "mmXQuantMode attr can not be null."), return ge::GRAPH_FAILED);
        auto mmXQuantMode = *mmXQuantModePtr_;
        OP_TILING_CHECK(mmXQuantMode != PERTENSOR_QUANT_MODE,
            OP_LOGE(context_->GetNodeName(), "mmXQuantMode only support 1(pertensor mode)."), return ge::GRAPH_FAILED);
        // mmWeightQuantMode
        OP_TILING_CHECK(mmWeightQuantModePtr_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "mmWeightQuantMode attr can not be null."), return ge::GRAPH_FAILED);
        auto mmWeightQuantMode = *mmWeightQuantModePtr_;
        OP_TILING_CHECK(mmWeightQuantMode != PERTENSOR_QUANT_MODE,
            OP_LOGE(context_->GetNodeName(), "mmWeightQuantMode only support 1(pertensor mode)."),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::SetHcclTiling() const
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(gmmXDataType_) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(context_->GetNodeName(), "alltoAllvDataType is not found in HCCL_DATA_TYPE."), return ge::GRAPH_FAILED);
    auto alltoAllvDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(gmmXDataType_)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
        alltoAllvDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.a2avCcTiling failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmQuantTiling, 1);
} // namespace optiling