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
#include "allto_allv_grouped_mat_mul_tiling_base.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "context_util.h"
#include "allto_allv_grouped_mat_mul_no_quant_tiling.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
bool AlltoAllvGmmNoQuantTiling::IsCapable()
{
    // only support float16/bfloat16
    if (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_FLOAT16 &&
        context_->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_BF16) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvGmmNoQuantTiling is capable.");
    return true;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetPlatformInfo.");
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetPlatformInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeAttrsInfo.");
    if (GetCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoOpTiling.");
    if (CheckDType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(SetHcclTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);
    context_->SetTilingKey(GetTilingKey());
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivCoreNum_, aicCoreNum_, aivCoreNum_));
    OP_LOGD(context_->GetNodeName(), "end DoOpTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoLibApiTiling.");
    auto dTypeForMM = matmul_tiling::DataType::DT_FLOAT16;
    if (gmmXDataType_ == ge::DT_BF16) {
        dTypeForMM = matmul_tiling::DataType::DT_BF16;
    }
    OP_LOGD(context_->GetNodeName(), "gmmXDataType_ is %d, dTypeForMM is %d.", static_cast<int>(gmmXDataType_),
        static_cast<int>(dTypeForMM));
    MMTilingParams mmParams = { a_, h1_, n1_, &expertBaseM_, &expertBaseK_, &expertBaseN_ };
    OP_TILING_CHECK(CalMMTiling(mmParams) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GMM CalMMTiling failed."), return ge::GRAPH_FAILED);
    SetMMTilingParams setMmParams = { dTypeForMM, a_, h1_, n1_, expertBaseM_, expertBaseN_, 0 };
    if (a_ != 0) {
        OP_TILING_CHECK(SetMMTiling(setMmParams) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "GMM SetMMTiling failed."), return ge::GRAPH_FAILED);
    }
    if (hasSharedExpertFlag_) {
        mmParams = { bs_, h2_, n2_, &sharedExpertBaseM_, &sharedExpertBaseK_, &sharedExpertBaseN_ };
        OP_TILING_CHECK(CalMMTiling(mmParams) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "MM CalMMTiling failed."), return ge::GRAPH_FAILED);
        setMmParams = { dTypeForMM, bs_, h2_, n2_, sharedExpertBaseM_, sharedExpertBaseN_, 1 };
        OP_TILING_CHECK(SetMMTiling(setMmParams) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "MM SetMMTiling failed."), return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end DoLibApiTiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmNoQuantTiling::GetTilingKey() const
{
    uint32_t templateMmDType = ADD_TPL_FP16;
    bool tilingkeyMm = false;
    bool tilingekyGmmTrans = false;
    bool tilingekyMmTrans = false;
    if (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_FLOAT16) {
        templateMmDType = ADD_TPL_FP16;
    } else if (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_BF16) {
        templateMmDType = ADD_TPL_BP16;
    }
    if (hasSharedExpertFlag_) {
        tilingkeyMm = true;
    } else {
        tilingkeyMm = false;
    }
    if (tilingData->commonTilingInfo.isGmmWeightTrans) {
        tilingekyGmmTrans = true;
    } else {
        tilingekyGmmTrans = false;
    }
    if (tilingData->commonTilingInfo.isMmWeightTrans) {
        tilingekyMmTrans = true;
    } else {
        tilingekyMmTrans = false;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(templateMmDType, tilingkeyMm, tilingekyGmmTrans, tilingekyMmTrans);

    OP_LOGD(context_->GetNodeName(), "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "start GetWorkspaceSize.");
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "can not get workspace."),
        return ge::GRAPH_FAILED);
    uint64_t permuteOutSize = permuteOutFlag_ ? 0 : (a_ * h1_ * GetSizeByDataType(gmmXDataType_));
    workspaces[0] = libApiWorkSpaceSize_ + permuteOutSize;
    OP_LOGD(context_->GetNodeName(), "end GetWorkspaceSize.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "start PostTiling.");
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(context_->GetNodeName(), "tilingData is null!"),
        return ge::GRAPH_FAILED);
    tilingData->commonTilingInfo.epWorldSize = epWorldSize_;
    tilingData->commonTilingInfo.isGmmWeightTrans = transGmmWeight_;
    tilingData->commonTilingInfo.isMmWeightTrans = transMmWeight_;
    tilingData->commonTilingInfo.isPermuteOut = permuteOutFlag_;
    tilingData->commonTilingInfo.BSK = bsk_;
    tilingData->commonTilingInfo.H1 = h1_;
    tilingData->commonTilingInfo.E_ep = e_;
    tilingData->commonTilingInfo.N1 = n1_;
    tilingData->commonTilingInfo.A = a_;
    tilingData->commonTilingInfo.H2 = h2_;
    tilingData->commonTilingInfo.N2 = n2_;
    tilingData->commonTilingInfo.BS = bs_;
    tilingData->commonTilingInfo.isSendCntsTensor = false;
    tilingData->commonTilingInfo.isRecvCntsTensor = false;
    tilingData->commonTilingInfo.aicCoreNum = aicCoreNum_;
    tilingData->commonTilingInfo.aivCoreNum = aivCoreNum_;
    tilingData->commonTilingInfo.commOut = 0;
    tilingData->commonTilingInfo.isNeedMM = hasSharedExpertFlag_;
    // set sendCnt
    errno_t ret = memcpy_s(&(tilingData->aicpuTiling.sendCnt), EXPERT_MAX_VALUE * sizeof(int64_t),
        sendCountsPtr_->GetData(), sendCountsPtr_->GetSize() * sizeof(int64_t));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d.", ret);
        return ge::GRAPH_FAILED;
    }
    // set recvCnt
    ret = memcpy_s(&(tilingData->aicpuTiling.recvCnt), EXPERT_MAX_VALUE * sizeof(int64_t), recvCountsPtr_->GetData(),
        recvCountsPtr_->GetSize() * sizeof(int64_t));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d.", ret);
        return ge::GRAPH_FAILED;
    }

    // print tiling info
    PrintCommonTilingInfo(tilingData->commonTilingInfo);
    PrintMatmulTilingData(tilingData->gmmTilingData, "gmm");
    if (hasSharedExpertFlag_) {
        PrintMatmulTilingData(tilingData->mmTilingData, "mm");
    }
    OP_LOGD(context_->GetNodeName(), "end PostTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::CheckDType() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckDType.");
    OP_TILING_CHECK((context_->GetInputDesc(GMM_X_INDEX) == nullptr) || (context_->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(context_->GetNodeName(), "GetInputDesc gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "GetOutputDesc y returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((context_->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
        (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_BF16),
        OP_LOGE(context_->GetNodeName(), "Unsupported dataType, gmmx only support float16 and bfloat16!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((context_->GetInputDesc(GMM_X_INDEX)->GetDataType() !=
        context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType()) ||
        (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() !=
        context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType()),
        OP_LOGE(context_->GetNodeName(), "The dataType of gmmWeight and gmmY should be the same with gmmX."),
        return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        auto mmXDex = context_->GetOptionalInputDesc(MM_X_INDEX);
        OP_TILING_CHECK(mmXDex == nullptr, OP_LOGE(context_->GetNodeName(), "Flag isNeedMM is True, but MM_X is null."),
            return ge::GRAPH_FAILED);
        auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX);
        OP_TILING_CHECK(mmWeightDesc == nullptr,
            OP_LOGE(context_->GetNodeName(), "Flag isNeedMM is True, MM_WEIGHT is null."), return ge::GRAPH_FAILED);
        auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(mmYDesc == nullptr, OP_LOGE(context_->GetNodeName(), "GetOutputDesc mmY returned null."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
            (context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_BF16),
            OP_LOGE(context_->GetNodeName(), "Unsupported dataType, mmx only support float16 and bfloat16!"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
            context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType()) ||
            (context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType()),
            OP_LOGE(context_->GetNodeName(), "The dataType of mmWeight and mmY should be the same with mmX."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != context_->GetInputDesc(GMM_X_INDEX)->GetDataType(),
            OP_LOGE(context_->GetNodeName(), "mmX data type (%s) must be the same as gmmX data type (%s) when shared expert is enabled.",
                ge::TypeUtils::DataTypeToSerialString(context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType()).c_str(),
                ge::TypeUtils::DataTypeToSerialString(context_->GetInputDesc(GMM_X_INDEX)->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType() != context_->GetInputDesc(GMM_X_INDEX)->GetDataType(),
            OP_LOGE(context_->GetNodeName(), "mmWeight data type (%s) must be the same as gmmX data type (%s) when shared expert is enabled.",
                ge::TypeUtils::DataTypeToSerialString(context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType()).c_str(),
                ge::TypeUtils::DataTypeToSerialString(context_->GetInputDesc(GMM_X_INDEX)->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckDType.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::SetHcclTiling() const
{
    OP_LOGD(context_->GetNodeName(), "start SetHcclTiling.");
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(context_->GetNodeName(), "Tiling Data is null!"),
        return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    auto inputDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(context_->GetNodeName(), "%s is Unsupported outputdata type!",
        Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(inputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(context_->GetNodeName(), "%s is Unsupported inputdata type!",
        Ops::Base::ToString(inputDataType).c_str()),
        return ge::GRAPH_FAILED);

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(inputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType,
        alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclInitTiling) != 0,
        OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->alltoAllvCcTiling) != 0,
        OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling alltoAllvCcTiling failed"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end SetHcclTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::CalMMTiling(MMTilingParams &params) const
{
    OP_LOGD(context_->GetNodeName(), "start CalMMTiling.");
    uint32_t tempBaseN = BEST_BASE_N;
    while (tempBaseN > static_cast<uint32_t>(params.curMaxN)) {
        tempBaseN = tempBaseN >> 1;
    }
    if (tempBaseN < static_cast<uint32_t>(params.curMaxN)) {
        tempBaseN = tempBaseN << 1;
    }
    *params.curBaseN = std::min<int32_t>(BEST_BASE_N, tempBaseN);

    OP_TILING_CHECK(*params.curBaseN == 0, OP_LOGE(context_->GetNodeName(), "curBaseN should not be 0."),
        return ge::GRAPH_FAILED);

    // 基于使能double buffer的L0B内存计算baseK
    *params.curBaseK =
        (l0bSize_ / DOUBLE_BUFFER) / (*params.curBaseN * GetSizeByDataType(gmmXDataType_)); // 相关*怎么处理 未知
    *params.curBaseK = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(*params.curBaseK)));
    if (*params.curBaseK > MAX_BASE_K) {
        *params.curBaseK = MAX_BASE_K;
        int32_t maxBaseN =
            SixteenAlign(l0bSize_ / DOUBLE_BUFFER / (*params.curBaseK * GetSizeByDataType(gmmXDataType_)));
        *params.curBaseN = std::min<int32_t>(*params.curBaseN, maxBaseN);
        *params.curBaseN = std::max<int32_t>(CUBE_BLOCK,
            SixteenAlign(static_cast<uint32_t>(*params.curBaseN), true)); // 16: minimum value for baseN
    }
    if (*params.curBaseK > params.curMaxK) {
        *params.curBaseK =
            std::min<int32_t>(*params.curBaseK, SixteenAlign(static_cast<uint32_t>(params.curMaxK), true));
    }
    OP_TILING_CHECK(*params.curBaseK == 0, OP_LOGE(context_->GetNodeName(), "curBaseK should not be 0."),
        return ge::GRAPH_FAILED);
    // 基于使能double buffer的L0A内存和L0B内存计算baseM(cube)
    uint32_t maxBaseM = l0cSize_ / (*params.curBaseN * sizeof(float));
    *params.curBaseM = std::min<uint32_t>(
        (l0aSize_ / DOUBLE_BUFFER) / (*params.curBaseK * GetSizeByDataType(gmmXDataType_)), maxBaseM);
    *params.curBaseM = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(*params.curBaseM)));
    if (params.curMaxM != 0 && *params.curBaseM > params.curMaxM) {
        *params.curBaseM = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(params.curMaxM), true));
    }
    OP_TILING_CHECK(*params.curBaseM == 0, OP_LOGE(context_->GetNodeName(), "curBaseM should not be 0."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CalMMTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmNoQuantTiling::SetMMTiling(SetMMTilingParams &params) const
{
    OP_LOGD(context_->GetNodeName(), "start SetMMTiling.");
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, params.matmulDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, params.matmulDtype, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, params.matmulDtype);
    mm.SetOrgShape(params.curMaxM, params.curMaxN, params.curMaxK);
    mm.SetShape(params.curMaxM, params.curBaseN, params.curMaxK);
    mm.SetFixSplit(std::min(params.curBaseM, params.curMaxM), params.curBaseN);
    mm.SetBufferSpace(l1Size_, l0cSize_, ubSize_);
    if (params.type == 0) {
        OP_TILING_CHECK(mm.GetTiling(tilingData->gmmTilingData) == -1,
            OP_LOGE(context_->GetNodeName(), "gmm matmul getTiling failed."), return ge::GRAPH_FAILED);
    } else if (params.type == 1) {
        OP_TILING_CHECK(mm.GetTiling(tilingData->mmTilingData) == -1,
            OP_LOGE(context_->GetNodeName(), "mm matmul getTiling failed."), return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end SetMMTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvGmmNoQuantTiling::PrintMatmulTilingData(::TCubeTiling msg, const std::string &tilingType)
{
    std::stringstream ss;
    ss << tilingType << ": "
       << "usedCoreNum=" << msg.usedCoreNum << ", "
       << "M=" << msg.M << ", "
       << "N=" << msg.N << ", "
       << "Ka=" << msg.Ka << ", "
       << "Kb=" << msg.Kb << ", "
       << "singleCoreM=" << msg.singleCoreM << ", "
       << "singleCoreN=" << msg.singleCoreN << ", "
       << "singleCoreK=" << msg.singleCoreK << ", "
       << "baseM=" << msg.baseM << ", "
       << "baseN=" << msg.baseN << ", "
       << "baseK=" << msg.baseK << ", "
       << "stepKa=" << msg.stepKa << ", "
       << "stepKb=" << msg.stepKb << ", "
       << "stepM=" << msg.stepM << ", "
       << "stepN=" << msg.stepN << ", "
       << "isBias=" << msg.isBias << ", "
       << "transLength=" << msg.transLength << ", "
       << "iterateOrder=" << msg.iterateOrder << ", "
       << "dbL0A=" << msg.dbL0A << ", "
       << "dbL0B=" << msg.dbL0B << ", "
       << "dbL0C=" << msg.dbL0C << ", "
       << "shareMode=" << msg.shareMode << ", "
       << "shareL1Size=" << msg.shareL1Size << ", "
       << "shareL0CSize=" << msg.shareL0CSize << ", "
       << "shareUbSize=" << msg.shareUbSize << ", "
       << "batchM=" << msg.batchM << ", "
       << "batchN=" << msg.batchN << ", "
       << "singleBatchM=" << msg.singleBatchM << ", "
       << "singleBatchN=" << msg.singleBatchN;
    OP_LOGI(context_->GetNodeName(), " %s", ss.str().c_str());
}

void AlltoAllvGmmNoQuantTiling::PrintCommonTilingInfo(AlltoAllvGmmCommonTilingInfo &commonTilingInfo) const
{
    std::stringstream ss;
    ss << "commonTilingInfo: "
       << "BSK=" << commonTilingInfo.BSK << ", "
       << "BS=" << commonTilingInfo.BS << ", "
       << "H1=" << commonTilingInfo.H1 << ", "
       << "H2=" << commonTilingInfo.H2 << ", "
       << "A=" << commonTilingInfo.A << ", "
       << "N1=" << commonTilingInfo.N1 << ", "
       << "N2=" << commonTilingInfo.N2 << ", "
       << "epWorldSize_=" << commonTilingInfo.epWorldSize << ", "
       << "E_ep=" << commonTilingInfo.E_ep << ", "
       << "commOut=" << commonTilingInfo.commOut << ", "
       << "aivCoreNum=" << commonTilingInfo.aivCoreNum << ", "
       << "aicCoreNum=" << commonTilingInfo.aicCoreNum << ", "
       << "isGmmWeightTrans=" << commonTilingInfo.isGmmWeightTrans << ", "
       << "isMmWeightTrans=" << commonTilingInfo.isMmWeightTrans << ", "
       << "isSendCntsTensor=" << commonTilingInfo.isSendCntsTensor << ", "
       << "isRecvCntsTensor=" << commonTilingInfo.isRecvCntsTensor << ", "
       << "isPermuteOut=" << commonTilingInfo.isPermuteOut << ", "
       << "isNeedMM=" << commonTilingInfo.isNeedMM;
    OP_LOGI(context_->GetNodeName(), " %s", ss.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmNoQuantTiling, 0);
} // namespace optiling