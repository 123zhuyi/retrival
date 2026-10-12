/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <type_traits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferDataType4Range(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4Range");
    DataType startDataType = context->GetInputDataType(0);
    DataType limitDataType = context->GetInputDataType(1);
    DataType deltaDataType = context->GetInputDataType(2);
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_CHECK_IF(
        (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Cannot get platform info!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "soc version is %s", platform_info.str_info.short_soc_version.c_str());
    if ((startDataType == limitDataType) && (limitDataType == deltaDataType)) {
        if (startDataType == ge::DT_DOUBLE && platform_info.str_info.short_soc_version == "Ascend950") {
            context->SetOutputDataType(0, ge::DT_FLOAT);
        } else {
            context->SetOutputDataType(0, startDataType);
        }
    } else {
        context->SetOutputDataType(0, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Range).InferDataType(InferDataType4Range);

} // namespace ops