/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shape_plugin.cpp
 * \brief
 */

#include <cinttypes>
#include <cstdint>
#include <exception>
#include <limits>

#include "nlohmann/json.hpp"
#include "onnx_common.h"
#include "conversion/strided_slice_v2/op_graph/strided_slice_v2_proto.h"
#include "math/shape/op_graph/shape_proto.h"

using json = nlohmann::json;

namespace domi {

static Status ParseParamsShape(const ge::Operator& op_src, ge::Operator& op_dest)
{
    const std::string op_name = GetOpName(op_dest);
    ge::AscendString source_name_string;
    const std::string source_name = op_src.GetName(source_name_string) == ge::GRAPH_SUCCESS ?
                                        source_name_string.GetString() :
                                        std::string();
    const std::string node_name = source_name.empty() ? op_name : source_name;
    ge::AscendString source_type;
    if (op_src.GetOpType(source_type) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op_name.c_str(), "Get Shape source operator type failed: node_name[%s]", node_name.c_str());
        return FAILED;
    }
    const std::string original_type = source_type.GetString();
    OP_LOGD(op_name.c_str(), "ParseParamsShape begin: node_name[%s], op_type[%s]", node_name.c_str(),
            original_type.c_str());

    // Shape-15 and later define optional start/end attributes. Keep the ONNX
    // defaults when an attribute is not present in the source node.
    int64_t start = 0;
    int64_t end = std::numeric_limits<int64_t>::max();
    bool has_start_attr = false;
    bool has_end_attr = false;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            const json attrs = json::parse(attrs_string.GetString());
            const json::const_iterator attribute_it = attrs.find("attribute");
            if (attribute_it == attrs.end() || !attribute_it->is_array()) {
                OP_LOGE(op_name.c_str(), "Shape source attribute JSON has no attribute array.");
                return FAILED;
            }

            for (const json& attr : *attribute_it) {
                const json::const_iterator name_it = attr.find("name");
                if (name_it == attr.end() || !name_it->is_string()) {
                    OP_LOGW(op_name.c_str(), "Shape source attribute has no valid name.");
                    continue;
                }

                const std::string attr_name = name_it->get<std::string>();
                if (attr_name != "start" && attr_name != "end") {
                    continue;
                }

                const json::const_iterator type_it = attr.find("type");
                const json::const_iterator value_it = attr.find("i");
                const int expected_type = static_cast<int>(ge::onnx::AttributeProto::INT);
                if (type_it == attr.end() || !type_it->is_number_integer() || type_it->get<int>() != expected_type ||
                    value_it == attr.end() || !value_it->is_number_integer()) {
                    OP_LOGW(op_name.c_str(), "Shape %s attr has unexpected JSON type or value.", attr_name.c_str());
                    continue;
                }

                const int64_t value = value_it->get<int64_t>();
                if (attr_name == "start") {
                    start = value;
                    has_start_attr = true;
                } else {
                    end = value;
                    has_end_attr = true;
                }
            }
        } catch (const std::exception& exception) {
            OP_LOGE(op_name.c_str(), "Parse Shape source attribute JSON failed: %s", exception.what());
            return FAILED;
        }
    }

    OP_LOGD(op_name.c_str(),
            "Shape attributes resolved: node_name[%s], start[%" PRId64 "](%s), end[%" PRId64 "](%s), "
            "dtype[DT_INT64]",
            node_name.c_str(), start, has_start_attr ? "provided" : "default", end,
            has_end_attr ? "provided" : "default");
    // PartitionedCall is a generic parser-side type, so its ONNX I/O ports
    // must be declared explicitly before the parser connects model tensors.
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("dtype", static_cast<uint32_t>(ge::DT_INT64));
    // ExpandOneToManyGraph looks up the callback by the framework original type.
    // The ONNX parser passes the versioned type in the source Operator type.
    op_dest.SetAttr("original_type", original_type);
    op_dest.SetAttr("name", node_name);
    op_dest.SetAttr("start", start);
    op_dest.SetAttr("end", end);
    return SUCCESS;
}

static ge::Operator MakeShapeSliceConst(const std::string& name, int64_t value)
{
    std::vector<int64_t> values{value};
    std::vector<int64_t> dims{1};
    ge::Tensor tensor = Vec2Tensor(values, dims, ge::DT_INT64);
    return ge::op::Const(name.c_str()).set_attr_value(tensor);
}

/*
 * Parser-side one-to-many expansion:
 *
 *   PartitionedCall
 *   (original ONNX Shape(X, start, end))
 *        |
 *        v
 *   Data(X) --> Shape --> StridedSliceV2 --> Y
 *                          ^   ^   ^   ^
 *                          +---+---+---+
 *                       Const(start/end/axes/strides)
 */
static Status ParseOpToGraphShape(const ge::Operator& op, ge::Graph& graph)
{
    const std::string op_name = GetOpName(op);
    std::string node_name;
    if (op.GetAttr("name", node_name) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op_name.c_str(), "Get Shape node name failed.");
        return FAILED;
    }

    int64_t start = 0;
    int64_t end = std::numeric_limits<int64_t>::max();
    if (op.GetAttr("start", start) != ge::GRAPH_SUCCESS || op.GetAttr("end", end) != ge::GRAPH_SUCCESS) {
        OP_LOGE(op_name.c_str(), "Get Shape start/end attributes failed: node_name[%s]", node_name.c_str());
        return FAILED;
    }
    ge::Operator data = ge::op::Data((node_name + "_Data").c_str()).set_attr_index(0);
    ge::Operator shape = ge::op::Shape((node_name + "_Shape").c_str()).set_input_x(data).set_attr_dtype(ge::DT_INT64);
    ge::Operator begin = MakeShapeSliceConst(node_name + "_ShapeBegin", start);
    ge::Operator end_op = MakeShapeSliceConst(node_name + "_ShapeEnd", end);
    ge::Operator axes = MakeShapeSliceConst(node_name + "_ShapeAxes", 0);
    ge::Operator strides = MakeShapeSliceConst(node_name + "_ShapeStrides", 1);
    ge::Operator slice = ge::op::StridedSliceV2((node_name + "_StridedSliceV2").c_str())
                             .set_input_x(shape)
                             .set_input_begin(begin)
                             .set_input_end(end_op)
                             .set_input_axes(axes)
                             .set_input_strides(strides);
    std::vector<ge::Operator> inputs{data};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(slice, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    OP_LOGD(op_name.c_str(),
            "ParseOpToGraphShape end: data[%s], shape[%s], slice[%s], begin[%" PRId64 "], end[%" PRId64
            "], axes[0], strides[1]",
            (node_name + "_Data").c_str(), (node_name + "_Shape").c_str(), (node_name + "_StridedSliceV2").c_str(),
            start, end);
    return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Shape"), ge::AscendString("ai.onnx::9::Shape"),
                   ge::AscendString("ai.onnx::10::Shape"), ge::AscendString("ai.onnx::11::Shape"),
                   ge::AscendString("ai.onnx::12::Shape"), ge::AscendString("ai.onnx::13::Shape"),
                   ge::AscendString("ai.onnx::14::Shape"), ge::AscendString("ai.onnx::15::Shape"),
                   ge::AscendString("ai.onnx::16::Shape"), ge::AscendString("ai.onnx::17::Shape"),
                   ge::AscendString("ai.onnx::18::Shape")})
    .ParseParamsByOperatorFn(ParseParamsShape)
    .ParseOpToGraphFn(ParseOpToGraphShape)
    .ImplyType(ImplyType::GELOCAL);
} // namespace domi
