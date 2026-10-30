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
 * \file cumsum.cc
 * \brief cumsum tiling
 */

#include "cumsum_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "cumsum_tiling_arch35.h"
#include "cumsum_tiling_ascendc_arch35.h"
#include "cumsum_tiling_ascendc_int_arch35.h"
#include "platform/platform_info.h"
#include "op_host/tiling_base_class.h"

using namespace ge;

namespace optiling {
static ge::graphStatus TilingCumsumForAscendc(gert::TilingContext* context)
{
    auto input = context->GetInputDesc(0);
    OP_CHECK_IF((input == nullptr), OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (input->GetDataType() != ge::DT_FLOAT && input->GetDataType() != ge::DT_FLOAT16 &&
         input->GetDataType() != ge::DT_BF16 && input->GetDataType() != ge::DT_INT32 &&
         input->GetDataType() != ge::DT_INT64 && input->GetDataType() != ge::DT_INT8 &&
         input->GetDataType() != ge::DT_UINT8 && input->GetDataType() != ge::DT_UINT64),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input",
            Ops::Base::ToString(input->GetDataType()).c_str(),
            "The dtype of input must be within the range [DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8, DT_UINT64]."),
        return ge::GRAPH_FAILED);
    if (input->GetDataType() == ge::DT_FLOAT || input->GetDataType() == ge::DT_FLOAT16 ||
        input->GetDataType() == ge::DT_BF16) {
        OP_LOGD(context->GetNodeName(), "TilingCumsumForAscendc simd, float type.");
        return TilingCumsumAscendc(context);
    } else {
        OP_LOGD(context->GetNodeName(), "TilingCumsumForAscendc simd, int type.");
        return TilingCumsum4Int(context);
    }
}

static ge::graphStatus Tiling4Cumsum(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const CumsumCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return TilingCumsumForAscendc(context);
}

static ge::graphStatus TilingPrepare4CumsumAscendc(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4CumsumAscendc.");

    auto compileInfo = context->GetCompiledInfo<CumsumCompileInfo>();

    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->core_num <= 0), OP_LOGE(context->GetNodeName(), "core num is negative."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ub_size <= 0), OP_LOGE(context->GetNodeName(), "fail to get ub size."), return ge::GRAPH_FAILED);

    compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->clSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get cache line size."),
        return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->blockSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get block size."),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        (compileInfo->vRegSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get vReg size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4CumsumAscendc.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Cumsum(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<CumsumCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD("TilingPrepare4Cumsum", "Ascend C TilingPrepare4Cumsum success.");
    return TilingPrepare4CumsumAscendc(context);
}

// register tiling interface of the Cumsum op.
IMPL_OP_OPTILING(Cumsum)
    .Tiling(Tiling4Cumsum)
    .TilingParse<CumsumCompileInfo>(TilingPrepare4Cumsum)
    .TilingInputsDataDependency({1});
} // namespace optiling
