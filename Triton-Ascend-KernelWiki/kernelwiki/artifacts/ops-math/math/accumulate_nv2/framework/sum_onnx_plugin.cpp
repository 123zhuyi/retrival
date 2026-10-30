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

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsSum(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    uint32_t input_num = node->input_size();
    if (input_num < 1) {
        OP_LOGE(GetOpName(op_dest).c_str(), "input_num must be 1");
        return FAILED;
    }
    op_dest.DynamicInputRegister("x", input_num);
    op_dest.SetAttr("N", input_num);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AccumulateNV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::Sum"),
                 ge::AscendString("ai.onnx::9::Sum"),
                 ge::AscendString("ai.onnx::10::Sum"),
                 ge::AscendString("ai.onnx::11::Sum"),
                 ge::AscendString("ai.onnx::12::Sum"),
                 ge::AscendString("ai.onnx::13::Sum"),
                 ge::AscendString("ai.onnx::14::Sum"),
                 ge::AscendString("ai.onnx::15::Sum"),
                 ge::AscendString("ai.onnx::16::Sum"),
                 ge::AscendString("ai.onnx::17::Sum"),
                 ge::AscendString("ai.onnx::18::Sum")})
  .ParseParamsFn(ParseParamsSum)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
