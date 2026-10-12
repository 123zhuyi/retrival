/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator.h"
#include "math_onnx_plugin_util.h"
#include "log/log.h"
#include "register/register.h"
#include "nlohmann/json.hpp"

namespace domi {
using json = nlohmann::json;

static void ApplyReverseSequenceAttr(ge::Operator& op_dest, const json& attr)
{
    if (attr.value("name", "") == "batch_axis" && attr.contains("i")) {
        op_dest.SetAttr("batch_dim", attr["i"].get<int>() == 1 ? 1 : 0);
    }
    if (attr.value("name", "") == "time_axis" && attr.contains("i")) {
        op_dest.SetAttr("seq_dim", attr["i"].get<int>());
    }
}

static Status ParseReverseSequenceAttrs(ge::Operator& op_dest, const ge::AscendString& attrs_string)
{
    try {
        json attrs = json::parse(attrs_string.GetString());
        if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
            for (const json& attr : attrs["attribute"]) {
                ApplyReverseSequenceAttr(op_dest, attr);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseParamsReverseSequence(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // set batch_dim's default value to 1, and set seq_dim's default value to 0
    op_dest.SetAttr("batch_dim", 1);
    op_dest.SetAttr("seq_dim", 0);

    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        return ParseReverseSequenceAttrs(op_dest, attrs_string);
    }
    return SUCCESS;
}
// register ReverseSequence op info to GE
REGISTER_CUSTOM_OP("ReverseSequence")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::10::ReverseSequence"), ge::AscendString("ai.onnx::11::ReverseSequence"),
                   ge::AscendString("ai.onnx::12::ReverseSequence"), ge::AscendString("ai.onnx::13::ReverseSequence"),
                   ge::AscendString("ai.onnx::14::ReverseSequence"), ge::AscendString("ai.onnx::15::ReverseSequence"),
                   ge::AscendString("ai.onnx::16::ReverseSequence"), ge::AscendString("ai.onnx::17::ReverseSequence"),
                   ge::AscendString("ai.onnx::18::ReverseSequence")})
    .ParseParamsByOperatorFn(ParseParamsReverseSequence)
    .ImplyType(ImplyType::TVM);
} // namespace domi
