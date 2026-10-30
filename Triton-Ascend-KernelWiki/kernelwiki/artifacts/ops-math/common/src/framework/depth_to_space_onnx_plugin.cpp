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
static Status ParseParamsDepthToSpace(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int block_size = 0;
    std::string mode = "DCR";
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "blocksize" && attr.type() == ge::onnx::AttributeProto::INT) {
            block_size = attr.i();
        } else if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            mode = attr.s();
        }
    }
    const int support_block_size = 2;
    if (block_size < support_block_size) {
        OP_LOGE(GetOpName(op_dest).c_str(), "blocksize, specifying the size of the spatial block, should be >=2, ");
        return FAILED;
    }

    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        return FAILED;
    }

    op_dest.SetAttr("block_size", block_size);
    op_dest.SetAttr("mode", mode);
    // set attr data format to NCHW.
    std::string output_format = "NCHW";
    op_dest.SetAttr("data_format", output_format);
    return SUCCESS;
}
// register DepthToSpace op info to GE
REGISTER_CUSTOM_OP("DepthToSpace")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::DepthToSpace"), ge::AscendString("ai.onnx::9::DepthToSpace"),
         ge::AscendString("ai.onnx::10::DepthToSpace"), ge::AscendString("ai.onnx::11::DepthToSpace"),
         ge::AscendString("ai.onnx::12::DepthToSpace"), ge::AscendString("ai.onnx::13::DepthToSpace"),
         ge::AscendString("ai.onnx::14::DepthToSpace"), ge::AscendString("ai.onnx::15::DepthToSpace"),
         ge::AscendString("ai.onnx::16::DepthToSpace"), ge::AscendString("ai.onnx::17::DepthToSpace"),
         ge::AscendString("ai.onnx::18::DepthToSpace")})
    .ParseParamsFn(ParseParamsDepthToSpace)
    .ImplyType(ImplyType::TVM);
} // namespace domi