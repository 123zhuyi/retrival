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
#include "conversion/pad_v3/op_graph/pad_v3_proto.h"
#include "conversion/mirror_pad/op_graph/mirror_pad_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
static Status parse_params_pad_v11(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::string mode_value = "constant";
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            mode_value = attr.s();
        }
    }
    op_dest.SetAttr("paddings_contiguous", false);
    op_dest.SetAttr("mode", mode_value);
    return SUCCESS;
}

static Status parse_params_pad_v9(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.SetAttr("name", node->name());
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("output", 1);
    op_dest.SetAttr("original_type", "ai.onnx::9::Pad");

    // attr max and min default value
    std::vector<int32_t> v_pads;
    bool set_pads_flag = false;
    float value = 0.0;
    std::string mode_value = "constant";
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            mode_value = attr.s();
        } else if (attr.name() == "pads") {
            set_pads_flag = true;
            unsigned int len = attr.ints_size();
            if (len & 1) {
                OP_LOGE(
                    GetOpName(op_dest).c_str(),
                    "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
                return FAILED;
            }
            unsigned int half = len / 2;
            for (unsigned int i = 0; i < half; i++) {
                v_pads.push_back(static_cast<int32_t>(attr.ints(i)));
                v_pads.push_back(static_cast<int32_t>(attr.ints(i + half)));
            }
        } else if (attr.name() == "value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            value = attr.f();
        }
    }

    if (!set_pads_flag) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int64_t> dims = {(int64_t)v_pads.size()};
    ge::Tensor tensor1 = Vec2Tensor(v_pads, dims, ge::DT_INT32);
    op_dest.SetAttr("paddings", tensor1);

    ge::Tensor scalar_const_value = CreateScalar(value, ge::DT_FLOAT);
    op_dest.SetAttr("constant_values", scalar_const_value);
    op_dest.SetAttr("mode", mode_value);
    return SUCCESS;
}

static Status ParseOpToGraphPad(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    ge::Tensor value1;
    if (op.GetAttr("paddings", value1) != SUCCESS) {
        return FAILED;
    }
    ge::Tensor value2;
    if (op.GetAttr("constant_values", value2) != SUCCESS) {
        return FAILED;
    }

    std::string mode = "constant";
    std::string mode_reflect = "REFLECT";
    op.GetAttr("mode", mode);
    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(value1);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    if (mode == "reflect") {
        auto mirror_pad = op::MirrorPad((ori_name + "_MirrorPad").c_str())
                              .set_input_x(data0)
                              .set_input_paddings(data1)
                              .set_attr_mode(mode_reflect);

        output_indexs.emplace_back(mirror_pad, vector<std::size_t>{0});
    } else {
        auto data2 = op::Const((ori_name + "_data2").c_str()).set_attr_value(value2);
        auto pad_v3 = op::PadV3((ori_name + "_PadV3").c_str())
                          .set_input_x(data0)
                          .set_input_paddings(data1)
                          .set_input_constant_values(data2)
                          .set_attr_mode(mode);

        output_indexs.emplace_back(pad_v3, vector<std::size_t>{0});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Pad"),
                   ge::AscendString("ai.onnx::9::Pad"),
                   ge::AscendString("ai.onnx::10::Pad")})
    .ParseParamsFn(parse_params_pad_v9)
    .ParseOpToGraphFn(ParseOpToGraphPad)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PadV3")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::Pad"),
                   ge::AscendString("ai.onnx::12::Pad"),
                   ge::AscendString("ai.onnx::13::Pad"),
                   ge::AscendString("ai.onnx::14::Pad"),
                   ge::AscendString("ai.onnx::15::Pad"),
                   ge::AscendString("ai.onnx::16::Pad"),
                   ge::AscendString("ai.onnx::17::Pad"),
                   ge::AscendString("ai.onnx::18::Pad")})
    .ParseParamsFn(parse_params_pad_v11)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
