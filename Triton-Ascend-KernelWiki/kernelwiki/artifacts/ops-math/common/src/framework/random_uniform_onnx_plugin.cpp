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
#include "math/muls/op_graph/muls_proto.h"
#include "math/shape/op_graph/shape_proto.h"

using namespace ge;
namespace domi {
static Status ParseParamsRandomuniform(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    const int input = 3;
    const int output = 1;
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);
    op_dest.SetAttr("original_type", "ai.onnx::11::RandomUniform");

    float high = 1.0;
    float low = 0.0;
    std::vector<int> op_shape;
    int seed = 0;
    int dtype = 1;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
            dtype = attr.i();
        } else if (attr.name() == "high" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            high = attr.f();
        } else if (attr.name() == "low" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            low = attr.f();
        } else if (attr.name() == "seed" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            seed = (int)attr.f();
        } else if (attr.name() == "shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                op_shape.push_back(attr.ints(i));
            }
        }
    }

    int num = op_shape.size();
    std::vector<int64_t> dims = {num};
    ge::Tensor tensor = Vec2Tensor(op_shape, dims, ge::DT_INT32);

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("shape", tensor);
    op_dest.SetAttr("max", high);
    op_dest.SetAttr("min", low);
    op_dest.SetAttr("seed", seed);
    op_dest.SetAttr("dtype", dtype);
    return SUCCESS;
}

namespace {
struct RandomuniformProp {
    std::string ori_name;
    float max_f = 0.0;
    float min_f = 0.0;
    int dtype = 1;
    int seed = 0;
    ge::Tensor shape;
};

Status GetProperty(const ge::Operator& op, RandomuniformProp& prop)
{
    if (op.GetAttr("name", prop.ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    if (op.GetAttr("shape", prop.shape) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get shape from op failed");
        return FAILED;
    }

    if (op.GetAttr("max", prop.max_f) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get max from op failed");
        return FAILED;
    }

    if (op.GetAttr("min", prop.min_f) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get min from op failed");
        return FAILED;
    }

    if (op.GetAttr("dtype", prop.dtype) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get dtype from op failed");
        return FAILED;
    }

    if (op.GetAttr("seed", prop.seed) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get seed from op failed");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

static Status ParseOpToGraphRandomuniform(const ge::Operator& op, ge::Graph& graph)
{
    RandomuniformProp prop;
    if (GetProperty(op, prop) != SUCCESS) {
        return FAILED;
    }
    auto data0 = op::Const((prop.ori_name + "_data0").c_str()).set_attr_value(prop.shape);
    // cast output to dst_dtype(onnx : Ascend)
    // float32, float16, int32, int64
    std::map<int, int> kvlist = {{1, 0}, {10, 1}, {6, 3}, {2, 9}};
    if (kvlist.find(prop.dtype) == kvlist.end()) {
        OP_LOGE(GetOpName(op).c_str(), "only support float32/float16/int32/int64, but got %d", prop.dtype);
        return FAILED;
    }
    ge::DataType temp_type = GetOmDtypeFromOnnxDtype(prop.dtype);

    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    if (static_cast<DataTypeOnnx>(prop.dtype) == DataTypeOnnx::DTO_INT32 ||
        static_cast<DataTypeOnnx>(prop.dtype) == DataTypeOnnx::DTO_UINT8) {
        int32_t max = prop.max_f;
        ge::Tensor scalar_max = CreateScalar(max, temp_type);
        auto data1 = op::Const((prop.ori_name + "_data1").c_str()).set_attr_value(scalar_max);

        int32_t min = prop.min_f;
        ge::Tensor scalar_min = CreateScalar(min, temp_type);
        auto data2 = op::Const((prop.ori_name + "_data2").c_str()).set_attr_value(scalar_min);

        auto random_int = op::RandomUniformInt((prop.ori_name + "_RandomUniformInt").c_str())
                              .set_input_shape(data0)
                              .set_input_min(data2)
                              .set_input_max(data1)
                              .set_attr_seed(prop.seed)
                              .set_attr_seed2(prop.seed);
        std::vector<ge::Operator> inputs{data0, data1, data2};
        outputs.emplace_back(random_int, std::vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(outputs);
    } else {
        auto random = op::RandomUniform((prop.ori_name + "_RandomUniform").c_str())
                          .set_input_shape(data0)
                          .set_attr_dtype(temp_type)
                          .set_attr_seed(prop.seed)
                          .set_attr_seed2(prop.seed);
        float mul_num = prop.max_f - prop.min_f;
        auto random_mul = op::Muls((prop.ori_name + "_Muls").c_str()).set_input_x(random).set_attr_value(mul_num);
        auto random_float =
            op::Adds((prop.ori_name + "_Adds").c_str()).set_input_x(random_mul).set_attr_value(prop.min_f);
        std::vector<ge::Operator> inputs{data0};
        outputs.emplace_back(random_float, std::vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::RandomUniform"), ge::AscendString("ai.onnx::9::RandomUniform"),
         ge::AscendString("ai.onnx::10::RandomUniform"), ge::AscendString("ai.onnx::11::RandomUniform"),
         ge::AscendString("ai.onnx::12::RandomUniform"), ge::AscendString("ai.onnx::13::RandomUniform"),
         ge::AscendString("ai.onnx::14::RandomUniform"), ge::AscendString("ai.onnx::15::RandomUniform"),
         ge::AscendString("ai.onnx::16::RandomUniform"), ge::AscendString("ai.onnx::17::RandomUniform"),
         ge::AscendString("ai.onnx::18::RandomUniform")})
    .ParseParamsFn(ParseParamsRandomuniform)
    .ParseOpToGraphFn(ParseOpToGraphRandomuniform)
    .ImplyType(ImplyType::TVM);
} // namespace domi
