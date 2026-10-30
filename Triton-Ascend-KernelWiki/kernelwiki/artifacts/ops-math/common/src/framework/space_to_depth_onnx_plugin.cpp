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
static Status ParseParamsSpaceToDepth(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int blocksize_val = -1;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "blocksize" && attr.type() == ge::onnx::AttributeProto::INT) {
            blocksize_val = attr.i();
        }
    }
    if (blocksize_val <= 0) {
        OP_LOGE(
            GetOpName(op_dest).c_str(), "Because input tensor shape is [N,C,H,W] and "
                                        "output tensor shape is [N, C*blocksize*blocksize, H/blocksize, W/blocksize],"
                                        "attr blocksize must be INT.");
        return FAILED;
    }

    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        return FAILED;
    }
    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS) {
        return FAILED;
    }

    op_dest.SetAttr("block_size", blocksize_val);
    op_dest.SetAttr("data_format", "NCHW");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("SpaceToDepth")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::SpaceToDepth"), ge::AscendString("ai.onnx::9::SpaceToDepth"),
         ge::AscendString("ai.onnx::10::SpaceToDepth"), ge::AscendString("ai.onnx::11::SpaceToDepth"),
         ge::AscendString("ai.onnx::12::SpaceToDepth"), ge::AscendString("ai.onnx::13::SpaceToDepth"),
         ge::AscendString("ai.onnx::14::SpaceToDepth"), ge::AscendString("ai.onnx::15::SpaceToDepth"),
         ge::AscendString("ai.onnx::16::SpaceToDepth"), ge::AscendString("ai.onnx::17::SpaceToDepth"),
         ge::AscendString("ai.onnx::18::SpaceToDepth")})
    .ParseParamsFn(ParseParamsSpaceToDepth)
    .ImplyType(ImplyType::TVM);
} // namespace domi
