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
 * \file moe_re_routing_re_tiling.cpp
 * \brief
 */
#include <iostream>
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "moe_re_routing_re_tiling.h"

namespace optiling {

constexpr int64_t FP32_PER_FP8 = 4;
constexpr uint64_t MOE_RE_ROUTING_RE_WITHOUT_SCALE = 200000;
constexpr uint64_t MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT = 200100;
constexpr uint64_t MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0 = 200200;

void MoeReRoutingReTiling::SplitCore()
{
    // 先按照rankNum分核
    int64_t blockFactor = Ops::Base::CeilDiv(rankNums_, coreNum_);
    blockNumR_ = Ops::Base::CeilDiv(rankNums_, blockFactor);
    blockFactorR_ = Ops::Base::CeilDiv(rankNums_, blockNumR_);
    // tokenSum_ == 0时需要判断
    if (tokenSum_ == 0) {
        blockNumR_ = 1;
    }
    int64_t newBlockNumR_ = blockNumR_ == 0 ? 1 : blockNumR_;
    auto blockNumE = std::min(coreNum_ / newBlockNumR_, expertNum_);
    blockFactorE_ = Ops::Base::CeilDiv(expertNum_, blockNumE);
    blockNumE_ = Ops::Base::CeilDiv(expertNum_, blockFactorE_);
    blockNum_ = blockNumR_ * blockNumE_;
    return;
}

ge::graphStatus MoeReRoutingReTiling::SplitUb()
{
    // 切分ub
    int64_t reserveUbSize = INDEX_UB_SIZE * DOUBLE_BUFFER * ge::GetSizeByDataType(expertDtype_);
    int64_t canUseUbSize = (ubSize_ - reserveUbSize) / DOUBLE_BUFFER;
    ubFactor_ = Ops::Base::CeilAlign(canUseUbSize, static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingReTiling::DoOpTiling()
{
    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "check param fail."),
        return ge::GRAPH_FAILED);
    SplitCore();
    if (SplitUb() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SplitUb Failed.");
        return ge::GRAPH_FAILED;
    }
    if (hasScale_) {
        tilingKey_ =
            scaleDtype_ == ge::DT_FLOAT ? MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT : MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0;
        if ((tokenDtype_ == ge::DT_FLOAT8_E4M3FN || tokenDtype_ == ge::DT_FLOAT8_E5M2) && scaleDtype_ == ge::DT_FLOAT) {
            tilingKey_ = MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0;
        }
    } else {
        tilingKey_ = MOE_RE_ROUTING_RE_WITHOUT_SCALE;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeReRoutingReTiling::GetTilingKey() const
{
    return tilingKey_;
}

void MoeReRoutingReTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "MoeReRoutingReTiling tilingData: tokenSize is %ld, scaleSize is %ld, rankNum is %ld,"
        "expertNum is %ld, blockNumR = %ld, blockFactorR = %ld, blockNumE = %ld, blockFactorE = %ld,"
        "coreNum is %ld, ubFactor is %ld, idxType_ = %ld, tilingKey is %ld",
        tilingData_.get_tokenSize(),
        tilingData_.get_scaleSize(),
        tilingData_.get_rankNum(),
        tilingData_.get_expertNum(),
        tilingData_.get_blockNumR(),
        tilingData_.get_blockFactorR(),
        tilingData_.get_blockNumE(),
        tilingData_.get_blockFactorE(),
        tilingData_.get_coreNum(),
        tilingData_.get_ubFactor(),
        tilingData_.get_idxType(),
        tilingKey_);
    return;
}

ge::graphStatus MoeReRoutingReTiling::PostTiling()
{
    tilingData_.set_tokenSize(tokenSize_);
    if ((tokenDtype_ == ge::DT_FLOAT8_E4M3FN || tokenDtype_ == ge::DT_FLOAT8_E5M2) && scaleDtype_ == ge::DT_FLOAT) {
        tilingData_.set_scaleSize(scaleSize_ * FP32_PER_FP8);
    } else {
        tilingData_.set_scaleSize(scaleSize_);
    }
    tilingData_.set_rankNum(rankNums_);
    tilingData_.set_expertNum(expertNum_);
    tilingData_.set_coreNum(blockNum_);
    tilingData_.set_blockNumR(blockNumR_);
    tilingData_.set_blockFactorR(blockFactorR_);
    tilingData_.set_blockNumE(blockNumE_);
    tilingData_.set_blockFactorE(blockFactorE_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_idxType(idxType_);

    context_->SetBlockDim(blockNum_);
    context_->SetTilingKey(GetTilingKey());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    workspaces[0] = static_cast<size_t>(RESERVE_WORKSPACE_BYTE);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeReRouting, MoeReRoutingReTiling, 10000);

}  // namespace optiling