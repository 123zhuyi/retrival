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
#include "op_math_proto_extend.h"

using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static const int ONNX_1D_ATTR_LEN = 1;

static void SetAllIntListValue(const ge::onnx::AttributeProto& attr, std::vector<int32_t>& int_list)
{
    for (auto i = 0; i < attr.ints_size(); ++i) {
        int_list.push_back(attr.ints(i));
    }
}

static void SetAttrListValue(const ge::onnx::AttributeProto& attr, std::vector<int32_t>& attr_list)
{
    if (attr.ints_size() == ONNX_1D_ATTR_LEN) {
        attr_list.push_back(1);
        attr_list.push_back(1);
        attr_list.push_back(attr.ints(0));
        attr_list.push_back(attr.ints(0));
    } else {
        for (auto i = 0; i < attr.ints_size(); ++i) {
            attr_list.push_back(attr.ints(i));
        }
    }
}

static Status ParseParamsDeformableOffsets(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("DeformableOffsets", "Dynamic cast op_src to NodeProto failed");
        return FAILED;
    }

    std::vector<int> strides = {};
    std::vector<int> pads = {};
    std::vector<int> dilations = {1, 1, 1, 1};
    std::vector<int> ksize = {};
    std::string data_format = "NCHW";
    int deformable_groups = 1;
    bool modulated = true;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            SetAttrListValue(attr, strides);
        } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            SetAttrListValue(attr, pads);
        } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
            dilations.clear();
            SetAttrListValue(attr, dilations);
        } else if (attr.name() == "ksize" && attr.type() == ge::onnx::AttributeProto::INTS) {
            SetAllIntListValue(attr, ksize);
        } else if (attr.name() == "data_format" && attr.type() == ge::onnx::AttributeProto::STRING) {
            data_format = attr.s();
        } else if (attr.name() == "deformable_groups" && attr.type() == ge::onnx::AttributeProto::INT) {
            deformable_groups = attr.i();
        } else if (attr.name() == "modulated" && attr.type() == ge::onnx::AttributeProto::INT) {
            modulated = attr.i();
        }
    }

    op_dest.SetAttr("strides", strides);
    op_dest.SetAttr("pads", pads);
    op_dest.SetAttr("dilations", dilations);
    op_dest.SetAttr("ksize", ksize);
    op_dest.SetAttr("data_format", data_format);
    op_dest.SetAttr("deformable_groups", deformable_groups);
    op_dest.SetAttr("modulated", modulated);

    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(op_dest, 1, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        OP_LOGE(GetOpName(op_dest).c_str(), "ChangeFormatFromOnnx failed.");
        return FAILED;
    }
    return SUCCESS;
}

// register DeformableOffsets op info to GE
REGISTER_CUSTOM_OP("DeformableOffsets")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::9::DeformableOffsets"), ge::AscendString("ai.onnx::10::DeformableOffsets"),
         ge::AscendString("ai.onnx::11::DeformableOffsets"), ge::AscendString("ai.onnx::12::DeformableOffsets"),
         ge::AscendString("ai.onnx::13::DeformableOffsets"), ge::AscendString("ai.onnx::14::DeformableOffsets"),
         ge::AscendString("ai.onnx::15::DeformableOffsets"), ge::AscendString("ai.onnx::16::DeformableOffsets")})
    .ParseParamsFn(ParseParamsDeformableOffsets)
    .ImplyType(ImplyType::TVM);
} // namespace domi