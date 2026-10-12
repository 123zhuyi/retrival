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
 * \file matmul_all_reduce_add_rms_norm_tiling.cpp
 * \brief
 */
#ifndef _MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#define _MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_

#include "matmul_all_reduce_add_rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using Ops::Transformer::OpTiling::TilingRegistry;
namespace optiling {
namespace {
constexpr char MRN[] = "MatmulAllReduceAddRmsNorm";
constexpr char IMRN[] = "InplaceMatmulAllReduceAddRmsNorm";
} // namespace
MMNTilingTransferHelper::MMNTilingTransferHelper(
    MatmulAllReduceAddRmsNormTiling& MatmulAllReduceAddRmsNormTiling, Mc2Tiling::MatmulAllReduce910TilingData& data)
    : MatmulAllReduceTiling910(
          MatmulAllReduceAddRmsNormTiling.context_, &MatmulAllReduceAddRmsNormTiling.mrnCtxInfo_.mmrCtxInfo, &data),
      tilingProcesser_(MatmulAllReduceAddRmsNormTiling)
{}
ge::graphStatus MMNTilingTransferHelper::GetShapeAttrsInfo()
{
    return MatmulAllReduceTilingBase::AnalyzeShapeAttr();
}

bool MatmulAllReduceAddRmsNormTiling::HasTail() const
{
    return hasTail_;
}
ge::graphStatus MatmulAllReduceAddRmsNormTiling::CheckMRNInput(const MRNCtxInfo& mrnCtxInfo)
{
    // x1和residual数据类型是否相同
    auto x1Type = mrnCtxInfo.mmrCtxInfo.x1->GetDataType();
    auto residualType = mrnCtxInfo.arnCtxInfo.x2->GetDataType();
    OP_TILING_CHECK(
        x1Type != residualType,
        VECTOR_INNER_ERR_REPORT_TILING(
            context_->GetNodeName(),
            "In the not quant scenario, type of x1 and residual"
            "should be same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MatmulAllReduceAddRmsNormTiling::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(helper_->DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(CommonAddResNormTiling::CheckAddRmsNormInput(context_, mrnCtxInfo_.arnCtxInfo));
    GE_ASSERT_GRAPH_SUCCESS(ContextTransfer::CheckMRNCtxInfo(context_, mrnCtxInfo_));
    GE_ASSERT_GRAPH_SUCCESS(CheckMRNInput(mrnCtxInfo_));
    hasTail_ = (tilingData_.matmulAllReduceTilingData.param.tailCnt != 0);
    AddRmsNormTilingInputFromMM addRmsNormTilingInputFromMm;
    addRmsNormTilingInputFromMm.m = helper_->tileMValue_;
    addRmsNormTilingInputFromMm.n = helper_->args_.nValue;
    addRmsNormTilingInputFromMm.x1Dtype = helper_->args_.geCType;
    GE_ASSERT_TRUE(context_->GetPlatformInfo() != nullptr);
    AddRMSNormTilingDepend addRmsNormTilingDepend = {
        context_->GetNodeName(),
        *context_->GetPlatformInfo(),
        mrnCtxInfo_.arnCtxInfo,
        addRmsNormTilingInputFromMm,
        true,
        false};

    AddRMSNormTilingOutput addRmsNormTilingOutput = {tilingData_.addRMSNormTileTilingData, tilingOutAddRmsNormTile_};

    GE_ASSERT_GRAPH_SUCCESS(CommonAddResNormTiling::Tiling4AddRmsNorm(addRmsNormTilingDepend, addRmsNormTilingOutput));
    tilingData_.addRmsNormTilingeKeyData.ARNKeyTile = tilingOutAddRmsNormTile_.tilingKey;
    tilingData_.addRmsNormTilingeKeyData.ARNNumBlocksTile = tilingOutAddRmsNormTile_.numBlocks;

    if (HasTail()) {
        addRmsNormTilingDepend.addRmsNormTilingInputFromMm.m = helper_->tailMValue_;
        AddRMSNormTilingOutput addRmsNormTilingOutputTail = {
            tilingData_.addRMSNormTailTilingData, tilingOutAddRmsNormTail_};
        GE_ASSERT_GRAPH_SUCCESS(
            CommonAddResNormTiling::Tiling4AddRmsNorm(addRmsNormTilingDepend, addRmsNormTilingOutputTail));
        tilingData_.addRmsNormTilingeKeyData.ARNKeyTail = tilingOutAddRmsNormTail_.tilingKey;
        tilingData_.addRmsNormTilingeKeyData.ARNNumBlocksTail = tilingOutAddRmsNormTail_.numBlocks;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MatmulAllReduceAddRmsNormTiling::GetShapeAttrsInfo()
{
    if (strcmp(context_->GetNodeType(), MRN) == 0) {
        GE_ASSERT_GRAPH_SUCCESS(ContextTransfer::AssembleMRNCtxInfoFromMRNCtx(context_, mrnCtxInfo_));
    } else if (strcmp(context_->GetNodeType(), IMRN) == 0) {
        GE_ASSERT_GRAPH_SUCCESS(ContextTransfer::AssembleIMRNCtxInfoFromIMRNCtx(context_, mrnCtxInfo_));
    } else {
        OP_LOGE(context_->GetNodeName(), "Unsupported node type %s", context_->GetNodeType());
        return ge::GRAPH_FAILED;
    }
    GE_ASSERT_NOTNULL(helper_);
    return helper_->GetShapeAttrsInfo();
}
ge::graphStatus MatmulAllReduceAddRmsNormTiling::GetPlatformInfo()
{
    return helper_->GetPlatformInfo();
}
ge::graphStatus MatmulAllReduceAddRmsNormTiling::DoLibApiTiling()
{
    return helper_->DoLibApiTiling();
}
bool MatmulAllReduceAddRmsNormTiling::IsCapable()
{
    return helper_->IsCapable();
}
MatmulAllReduceAddRmsNormTiling::MatmulAllReduceAddRmsNormTiling(gert::TilingContext* context)
    : TilingBaseClass(context)
{
    helper_ = std::move(std::unique_ptr<MMNTilingTransferHelper>(
        new (std::nothrow) MMNTilingTransferHelper(*this, tilingData_.matmulAllReduceTilingData)));
}

ge::graphStatus MatmulAllReduceAddRmsNormTiling::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(helper_->GetWorkspaceSize());
    const uint64_t mc2WorkSpace = helper_->myWorkSpaceSize_;
    GE_ASSERT_TRUE(mc2WorkSpace >= SYS_WORKSPACE_SIZE);
    if (HasTail()) {
        GE_ASSERT_TRUE(tilingOutAddRmsNormTile_.workSpaceSize == tilingOutAddRmsNormTail_.workSpaceSize);
    }
    // 系统空间用mc2申请的就好了， arn的减去这部分
    GE_ASSERT_TRUE(tilingOutAddRmsNormTile_.workSpaceSize >= SYS_WORKSPACE_SIZE);
    const auto arnWorkSpace = tilingOutAddRmsNormTile_.workSpaceSize - SYS_WORKSPACE_SIZE;
    const auto myWorkSpace = mc2WorkSpace + arnWorkSpace;
    OP_LOGI(helper_->opName_, " Workspace %lu with detail: mc2: %lu arn：%u", myWorkSpace, mc2WorkSpace, arnWorkSpace);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceAddRmsNormTiling::PostTiling()
{
    constexpr size_t tilingDataSize = sizeof(Mc2Tiling::MatmulAllReduceAddRmsNormTilingData);
    OP_LOGD(
        helper_->opName_, "final tiling data size: %zu and context capacity size: %zu ", tilingDataSize,
        context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    OP_TILING_CHECK(
        tilingDataSize % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            helper_->opName_,
            "tiling data size[%zu] not aligned to"
            " 8",
            tilingDataSize),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    helper_->PrintTilingData();
    auto numBlocksOfArn = static_cast<uint64_t>(tilingOutAddRmsNormTile_.numBlocks);
    if (HasTail()) {
        numBlocksOfArn = std::max(numBlocksOfArn, static_cast<uint64_t>(tilingOutAddRmsNormTail_.numBlocks));
    }
    OP_LOGI(
        helper_->opName_, "ctx block dim: %lu, mc2 block dim %lu, arn block dim %lu", helper_->args_.aicCoreNum,
        helper_->args_.aicCoreNum, numBlocksOfArn);
    // 当前mc2给的aicCoreNum是硬件规格的最大个数, numBlocksOfArn取了尾和非尾的最大值，最大值应该小于等于硬件规格的aiv num
    // 2代表aic和aiv个数规格定义是1比2的关系
    GE_ASSERT_TRUE(helper_->args_.aicCoreNum * 2 >= numBlocksOfArn);
    context_->SetBlockDim(helper_->args_.aicCoreNum);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = context_->SetScheduleMode(batch_mode);
    GE_ASSERT_GRAPH_SUCCESS(ret);

    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceAddRmsNormTiling::GetTilingKey() const
{
    const auto mc2_key = helper_->GetTilingKey();
    const auto my_key = mc2_key; // use mc2 key as mrn key
    OP_LOGI(
        helper_->opName_, " tilingKey %lu with detail: mc2_key: %lu arn_key tile：%u arn_key tail: %u", my_key, mc2_key,
        tilingOutAddRmsNormTile_.tilingKey, tilingOutAddRmsNormTail_.tilingKey);
    return my_key;
}

struct DefaultCompileInfo {
};
static ge::graphStatus DefaultTilingParseFunc(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
};
static ge::graphStatus DefaultTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}
IMPL_OP_OPTILING(MatmulAllReduceAddRmsNorm)
    .Tiling(DefaultTilingFunc)
    .TilingParse<DefaultCompileInfo>(DefaultTilingParseFunc);

REGISTER_OPS_TILING_TEMPLATE(MatmulAllReduceAddRmsNorm, MatmulAllReduceAddRmsNormTiling, 2);
} // namespace optiling
#endif // _MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_