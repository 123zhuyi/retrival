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
 * \file moe_finalize_routing_v2_grad_without_scale_tiling_arch35.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_tiling.h"

namespace optiling {

constexpr int64_t TILING_KEY_WITHOUT_SCALE_NOT_CUT_H = 10001;
constexpr int64_t TILING_KEY_WITHOUT_SCALE_CUT_H = 10002;

class MoeFinalizeRoutingV2GradWithoutScaleRegbase : public MoeFinalizeRoutingV2GradTiling
{
public:
    explicit MoeFinalizeRoutingV2GradWithoutScaleRegbase(gert::TilingContext* context)
        : MoeFinalizeRoutingV2GradTiling(context)
    {}
    ~MoeFinalizeRoutingV2GradWithoutScaleRegbase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingV2GradTiling::Reset(context);
    }

protected:
    ge::graphStatus CalcTilingKey() override;

    bool IsCapable() override
    {
        if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
        return !isScalesExist_;
    }

    ge::graphStatus CheckOptionalInputDtype() override;
};

ge::graphStatus MoeFinalizeRoutingV2GradWithoutScaleRegbase::CheckOptionalInputDtype()
{
    OP_CHECK_IF(
        ((scalesType_ != ge::DT_FLOAT) && (scalesType_ != ge::DT_BF16) && (scalesType_ != ge::DT_FLOAT16)),
        OP_LOGE(nodeName_, "scales dtype must be FLOAT or FLOAT16 or BFLOAT16."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (expandedXType_ != gradYType_), OP_LOGE(nodeName_, "expanded_x and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (expandedRowIdxType_ != ge::DataType::DT_INT32),
        OP_LOGE(nodeName_, "expanded_row_idx dtype only support int32."), return ge::GRAPH_FAILED);
    if (isBiasExist_) {
        OP_CHECK_IF(
            (biasType_ != gradYType_), OP_LOGE(nodeName_, "bias and grad_y dtype must be same."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (expertIdxType_ != expandedRowIdxType_),
            OP_LOGE(nodeName_, "expert_idx and expanded_row_idx dtype must be same."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradWithoutScaleRegbase::CalcTilingKey()
{
    int64_t expandedRowIdxUbSize = blockSize_;
    hiddenPrePart_ = (aicoreParams_.ubSize - expandedRowIdxUbSize) / blockSize_ * blockSize_ / gradYTypeByteSize_;
    if (hidden_ <= hiddenPrePart_) {
        tilingKey_ = TILING_KEY_WITHOUT_SCALE_NOT_CUT_H;
    } else {
        tilingKey_ = TILING_KEY_WITHOUT_SCALE_CUT_H;
    }

    OP_CHECK_IF(
        (hiddenPrePart_ <= 0), OP_LOGE(nodeName_, "hiddenPrePart_ is less than 0."),
        return ge::GRAPH_FAILED);

    if (hidden_ <= hiddenPrePart_) {
        hiddenInnerLoops_ = 0;
        hiddenLastPart_ = 0;
    } else {
        hiddenInnerLoops_ = hidden_ / hiddenPrePart_;
        hiddenLastPart_ = hidden_ % hiddenPrePart_;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeFinalizeRoutingV2Grad, MoeFinalizeRoutingV2GradWithoutScaleRegbase, 2000);

} // namespace optiling