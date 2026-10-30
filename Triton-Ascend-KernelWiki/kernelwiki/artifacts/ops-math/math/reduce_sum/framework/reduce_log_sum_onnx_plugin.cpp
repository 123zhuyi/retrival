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
#include "math/reduce_sum/op_graph/reduce_sum_proto.h"
#include "math/log/op_graph/log_proto.h"

using namespace ge;
using ge::Operator;

namespace domi {
static Status GetInputTensorDimNum(const Operator& data_op, int64_t& dim_num) {
  ge::TensorDesc input_desc = data_op.GetInputDesc(0);
  auto shape = input_desc.GetShape();
  if (shape.GetDimNum() <= 0) {
    OP_LOGE("GetInputTensorDimNum", "Get input shape is invalid.");
    return FAILED;
  }

  dim_num = shape.GetDimNum();
  OP_LOGI(GetOpName(data_op).c_str(), "GetInputTensorDimNum is: %ld", dim_num);
  return SUCCESS;
}

static Status parse_params_reduce_sum(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int input_size = node->input_size();
    int output_size = node->output_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("output", output_size);
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceLogSum");

    std::vector<int> v_axes = {};
    bool keep_dims_attr = true;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                v_axes.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims_attr = (attr.i() == 1);
        }
    }

    int num = v_axes.size();
    std::vector<int64_t> dims = {};
    if (num != 0) {
        dims.push_back(num);
    }
    ge::Tensor tensor = Vec2Tensor(v_axes, dims, ge::DT_INT32);

    op_dest.SetAttr("axes", tensor);
    op_dest.SetAttr("keep_dims", keep_dims_attr);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphReduceLogSum(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    ge::Tensor value;
    if (op.GetAttr("axes", value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get value from op failed");
        return FAILED;
    }

    if (value.GetSize() == 0) {
        int64_t input_dim_num = 0;
        if (GetInputTensorDimNum(op, input_dim_num) != SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "Failed to get input tensor dimensions");
            return FAILED;
        }
        std::vector<int64_t> v_axes;
        for (int64_t i = 0; i < input_dim_num; ++i) {
            v_axes.push_back(i);
        }
        int num = v_axes.size();
        std::vector<int64_t> dims = {};
        if (num != 0) {
            dims.push_back(num);
        }
        OP_LOGI(GetOpName(op).c_str(), "num is: %d, dims is: %d", num, dims.size());
        value = Vec2Tensor(v_axes, dims, ge::DT_INT64);
    }

    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(value);
    auto reducesum = op::ReduceSum((ori_name + "_ReduceSum").c_str()).set_input_x(data0).set_input_axes(data1);

    bool flag = false;
    if (op.GetAttr("keep_dims", flag) != SUCCESS) {
        ge::AscendString op_name;
        (void)op.GetName(op_name);
        OP_LOGE(op_name.GetString(), "get keep_dims from op failed");
        return FAILED;
    }
    reducesum.set_attr_keep_dims(flag);

    auto reducelogsum = op::Log((ori_name + "_Log").c_str()).set_input_x(reducesum);

    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    output_indexs.emplace_back(reducelogsum, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

static Status ParseParamsReduceLogSum13(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("ReduceLogSum13", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::13::ReduceLogSum");

    int input_size = node->input_size();
    bool keep_dims = true;
    int noop_with_empty_axes = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        } else if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT) {
            noop_with_empty_axes = attr.i();
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("input_size", input_size);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.SetAttr("noop_with_empty_axes", noop_with_empty_axes);
    return SUCCESS;
}

namespace {
struct ReduceLogSum13Prop {
    std::string ori_name;
    bool keep_dims = false;
    int input_num = 1;
    int empty_axes = 0;
};

Status GetProperty(const Operator& op, ReduceLogSum13Prop& prop)
{
    if (op.GetAttr("name", prop.ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    if (op.GetAttr("keep_dims", prop.keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }

    if (op.GetAttr("input_size", prop.input_num) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get input_num from op failed");
        return FAILED;
    }

    if (op.GetAttr("noop_with_empty_axes", prop.empty_axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attribute noop_with_empty_axes failed");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

static Status ParseOpToGraphReduceLogSum13(const Operator& op, Graph& graph)
{
    ReduceLogSum13Prop prop;
    if (GetProperty(op, prop) != SUCCESS) {
        return FAILED;
    }
    auto data0 = op::Data((prop.ori_name + "_data0").c_str()).set_attr_index(0);
    int num_input = 2;
    if (prop.input_num == 1 && prop.empty_axes == 0) {
        int64_t input_dim_num = 0;
        if (GetInputTensorDimNum(op, input_dim_num) != SUCCESS) {
          OP_LOGE(GetOpName(op).c_str(), "Failed to get input tensor dimensions");
          return FAILED;
        }

        std::vector<int64_t> v_axes;
        for (int64_t i = 0; i < input_dim_num; ++i) {
          v_axes.push_back(i);
        }
        ge::TensorDesc tensorDesc;
        std::vector<int64_t> dims = {input_dim_num};
        ge::Shape shape(dims);
        tensorDesc.SetShape(shape);
        tensorDesc.SetDataType(DT_INT64);
        ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_axes.data()), v_axes.size() * sizeof(int64_t));
        auto axes = op::Const((prop.ori_name + "_axes").c_str()).set_attr_value(tensor);
        std::vector<Operator> inputs{data0, axes};
        std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
        auto reducesum = op::ReduceSum((prop.ori_name + "_ReduceSum").c_str())
                             .set_input_x(data0)
                             .set_input_axes(axes)
                             .set_attr_keep_dims(prop.keep_dims)
                             .set_attr_noop_with_empty_axes(prop.empty_axes);
        auto reducelogsum = op::Log((prop.ori_name + "_Log").c_str()).set_input_x(reducesum);
        output_indexs.emplace_back(reducelogsum, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else if (prop.input_num == num_input) {
        auto data1 = op::Data((prop.ori_name + "_data1").c_str()).set_attr_index(1);
        auto reducesum13 = op::ReduceSum((prop.ori_name + "_ReduceSum").c_str())
                               .set_input_x(data0)
                               .set_input_axes(data1)
                               .set_attr_keep_dims(prop.keep_dims)
                               .set_attr_noop_with_empty_axes(prop.empty_axes);
        auto reducelogsum13 = op::Log((prop.ori_name + "_Log").c_str()).set_input_x(reducesum13);
        std::vector<Operator> inputs{data0, data1};
        std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
        output_indexs.emplace_back(reducelogsum13, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else {
        OP_LOGE(GetOpName(op).c_str(), "Input num or set attr is error");
        return FAILED;
    }
    return SUCCESS;
}

// register ReduceLogSum op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::ReduceLogSum"),
                 ge::AscendString("ai.onnx::9::ReduceLogSum"),
                 ge::AscendString("ai.onnx::10::ReduceLogSum"),
                 ge::AscendString("ai.onnx::11::ReduceLogSum"),
                 ge::AscendString("ai.onnx::12::ReduceLogSum")})
  .ParseParamsFn(parse_params_reduce_sum)
  .ParseOpToGraphFn(ParseOpToGraphReduceLogSum)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("ReduceLogSum")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::13::ReduceLogSum"),
                 ge::AscendString("ai.onnx::14::ReduceLogSum"),
                 ge::AscendString("ai.onnx::15::ReduceLogSum"),
                 ge::AscendString("ai.onnx::16::ReduceLogSum"),
                 ge::AscendString("ai.onnx::17::ReduceLogSum"),
                 ge::AscendString("ai.onnx::18::ReduceLogSum")})
  .ParseParamsFn(ParseParamsReduceLogSum13)
  .ParseOpToGraphFn(ParseOpToGraphReduceLogSum13)
  .ImplyType(ImplyType::TVM);
}  // namespace domi