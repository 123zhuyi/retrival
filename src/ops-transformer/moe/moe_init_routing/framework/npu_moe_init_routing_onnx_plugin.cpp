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
#include "nlohmann/json.hpp"

using namespace ge;
using json = nlohmann::json;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static const int REQ_ATTR_NUM = 1;

static Status ParseParamsNpuMoeInitRouting(const Message* op_src, ge::Operator& op_dest) {
    const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
    if (node == nullptr) {
      OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
    }
    // The initialization of required attributes count
    int req_attr_count = 0;
    for (const auto& attr: node->attribute()) {
      if (attr.name() == "active_num" && attr.type() == ge::onnx::AttributeProto::INT) {
        int active_num = attr.i();
        op_dest.SetAttr("active_num", active_num);
        ++req_attr_count;
      }
    }
    // Node must have required attribute active_num
    if (req_attr_count != REQ_ATTR_NUM) {
      OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr active_num.");
      return FAILED;
    }
    return SUCCESS;
}

// register npu_moe_init_routing op info to GE
REGISTER_CUSTOM_OP("MoeInitRouting")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::11::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::12::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::13::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::14::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::15::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::16::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::17::NPUMoeInitRouting"),
                 ge::AscendString("ai.onnx::18::NPUMoeInitRouting"),
                })
  .ParseParamsFn(ParseParamsNpuMoeInitRouting)
  .ImplyType(ImplyType::TVM);
} // namespace domi