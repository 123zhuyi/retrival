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
#include "op_math_proto_extend.h"
#include "math/arg_min/op_graph/arg_min_proto.h"
#include "platform/soc_spec.h"
#include <set>
#include <cstdlib>

extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

static bool IsRegBasePlatform()
{
    constexpr uint32_t maxLen = 32;
    char archStr[maxLen] = {0};
    if (rtGetSocSpec("version", "NpuArch", archStr, maxLen) != RT_ERROR_NONE) {
        OP_LOGD("ArgMin", "IsRegBasePlatform: get soc spec failed.");
        return false;
    }
    auto curArch = static_cast<NpuArch>(std::strtol(archStr, nullptr, 10));
    const static std::set<NpuArch> regbaseArch = {NpuArch::DAV_3510, NpuArch::DAV_5102};
    bool isRegbase = (regbaseArch.find(curArch) != regbaseArch.end());
    OP_LOGD("ArgMin", "IsRegBasePlatform: curArch=%u, is_regbase=%d.", static_cast<uint32_t>(curArch), isRegbase);
    return isRegbase;
}

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsArgMin(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    // 2.set original_type
    op_dest.SetAttr("original_type", "ai.onnx::11::ArgMin");

    // 3.set attr if needed
    int axis = 0;
    int keep_dims = 1;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis = attr.i();
            OP_LOGD(GetOpName(op_dest).c_str(), "set dimension = %d", axis);
        }
        if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = attr.i();
        }
        if (attr.name() == "select_last_index" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            OP_LOGE(GetOpName(op_dest).c_str(), "select_last_index should be 0, but now it's 1");
            return FAILED;
        }
    }

    op_dest.SetAttr("name", node->name());
    ge::Tensor scalar_axis = CreateScalar(axis, ge::DT_INT32);
    op_dest.SetAttr("dimension", scalar_axis);
    op_dest.SetAttr("keep_dims", keep_dims);
    return SUCCESS;
}

static Status ParseOpToGraphArgMin(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int keep_dims = 1;
    if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }
    ge::Tensor axis;
    if (op.GetAttr("dimension", axis) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get dimension from op failed");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

    auto const_op = op::Const((ori_name + "_const_data").c_str()).set_attr_value(axis);
    auto outputDtype = IsRegBasePlatform() ? ge::DT_INT64 : ge::DT_INT32;
    auto argMin = op::ArgMin((ori_name + "_ArgMin").c_str())
                      .set_input_x(data0)
                      .set_input_dimension(const_op)
                      .set_attr_dtype(outputDtype);

    if (keep_dims == 1) {
        auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(axis);

        auto expandDims = op::ExpandDims((ori_name + "_ExpandDims").c_str()).set_input_x(argMin).set_input_axis(data1);
        outputs.emplace_back(expandDims, vector<std::size_t>{0});
    } else {
        outputs.emplace_back(argMin, vector<std::size_t>{0});
    }

    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register ArgMin op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ArgMin"), ge::AscendString("ai.onnx::9::ArgMin"),
                   ge::AscendString("ai.onnx::10::ArgMin"), ge::AscendString("ai.onnx::11::ArgMin"),
                   ge::AscendString("ai.onnx::12::ArgMin"), ge::AscendString("ai.onnx::13::ArgMin"),
                   ge::AscendString("ai.onnx::14::ArgMin"), ge::AscendString("ai.onnx::15::ArgMin"),
                   ge::AscendString("ai.onnx::16::ArgMin"), ge::AscendString("ai.onnx::17::ArgMin"),
                   ge::AscendString("ai.onnx::18::ArgMin")})
    .ParseParamsFn(ParseParamsArgMin)
    .ParseOpToGraphFn(ParseOpToGraphArgMin)
    .ImplyType(ImplyType::TVM);
} // namespace domi
