/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"

namespace domi {
static Status AutoMappingByOpFnConcatExt(const ge::Operator& op_src, ge::Operator& op)
{
    std::vector<DynamicInputOutputInfo> dynamic_info;
    dynamic_info.emplace_back(kInput, "x", 1, "N", 1);
    return AutoMappingByOpFnDynamic(op_src, op, dynamic_info);
}

REGISTER_CUSTOM_OP("ConcatV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ConcatV2")
    .ParseParamsByOperatorFn(AutoMappingByOpFnConcatExt)
    .ImplyType(ImplyType::TVM);
} // namespace domi
