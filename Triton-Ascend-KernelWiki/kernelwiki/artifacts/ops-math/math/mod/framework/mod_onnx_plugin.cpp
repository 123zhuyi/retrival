/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "stub_ops.h"
#include "log/log.h"

namespace domi {

static Status ParseParamsMod(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "fmod") {
            OP_LOGW(GetOpName(op_dest).c_str(), "Current optype not surpport fmod, please ignore");
        }
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Mod")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::10::Mod"),
                   ge::AscendString("ai.onnx::11::Mod"),
                   ge::AscendString("ai.onnx::12::Mod"),
                   ge::AscendString("ai.onnx::13::Mod"),
                   ge::AscendString("ai.onnx::14::Mod"),
                   ge::AscendString("ai.onnx::15::Mod"),
                   ge::AscendString("ai.onnx::16::Mod"),
                   ge::AscendString("ai.onnx::17::Mod"),
                   ge::AscendString("ai.onnx::18::Mod")})
    .ParseParamsFn(ParseParamsMod)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
