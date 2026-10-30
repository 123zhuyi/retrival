/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mul_no_nan_tiling.cpp
 * \brief MulNoNan Host Tiling (atvoss BroadcastBaseTiling)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../op_kernel/mul_no_nan_dag.h"
#include "../op_kernel/mul_no_nan_struct.h"

namespace optiling {

using namespace ge;

static ge::graphStatus MulNoNanTilingFunc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE(context, "MulNoNan: GetInputDesc(0) returned nullptr");
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = inputDesc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;

    if (dtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<NsMulNoNan::MulNoNanCompute<half>::OpDag> brcTiling(context);
        ret = brcTiling.DoTiling();
        OP_CHECK_IF(
            ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "MulNoNan: BroadcastBaseTiling DoTiling failed for FP16"),
            return ret);
        context->SetTilingKey(GET_TPL_TILING_KEY(brcTiling.GetSchMode()));
    } else if (dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<NsMulNoNan::MulNoNanCompute<float>::OpDag> brcTiling(context);
        ret = brcTiling.DoTiling();
        OP_CHECK_IF(
            ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "MulNoNan: BroadcastBaseTiling DoTiling failed for FP32"),
            return ret);
        context->SetTilingKey(GET_TPL_TILING_KEY(brcTiling.GetSchMode()));
    } else if (dtype == ge::DT_BF16) {
        BroadcastBaseTiling<NsMulNoNan::MulNoNanCompute<bfloat16_t>::OpDag> brcTiling(context);
        ret = brcTiling.DoTiling();
        OP_CHECK_IF(
            ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "MulNoNan: BroadcastBaseTiling DoTiling failed for BF16"),
            return ret);
        context->SetTilingKey(GET_TPL_TILING_KEY(brcTiling.GetSchMode()));
    } else {
        OP_LOGE(context, "MulNoNan: unsupported dtype %d", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context, "MulNoNan: Tiling success");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMulNoNan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MulNoNanCompileInfo {};

IMPL_OP_OPTILING(MulNoNan).Tiling(MulNoNanTilingFunc).TilingParse<MulNoNanCompileInfo>(TilingParseForMulNoNan);

} // namespace optiling
