/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nlohmann/json.hpp"
#include "onnx_common.h"
#include "op_math_proto_extend.h"

using namespace ge;
using json = nlohmann::json;
namespace {
constexpr int INPUT_SIZE = 3;
constexpr int OUTPUT_SIZE = 1;
} // namespace
namespace domi {
static Status ParseParamsBernoulli(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // 1.add dynamic input and output
    op_dest.DynamicInputRegister("x", INPUT_SIZE);
    op_dest.DynamicOutputRegister("y", OUTPUT_SIZE);
    // 2.set original_type
    op_dest.SetAttr("original_type", "ai.onnx::15::Bernoulli");

    AscendString attrs_string;
    float seed = -1.0f;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        json attrs = json::parse(attrs_string.GetString());
        for (json& attr : attrs["attribute"]) {
            if (attr["name"] == "seed") {
                std::string seed_string = attr["f"];
                seed = atof(seed_string.c_str());
            } else if (attr["name"] == "dtype") {
                int dtype = attr["i"];
                ge::DataType out_dtype = GetOmDtypeFromOnnxDtype(dtype);
                op_dest.SetAttr("dtype", out_dtype);
            }
        }
    }

    int64_t om_seed = static_cast<int64_t>(seed);
    ge::Tensor scalar_seed = CreateScalar(om_seed, ge::DT_INT64);
    op_dest.SetAttr("seed", scalar_seed);
    op_dest.SetAttr("name", GetOpName(op_dest));

    return SUCCESS;
}

static Status ParseOpToGraphBernoulli(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    ge::DataType dtype = DT_UNDEFINED;
    op.GetAttr("dtype", dtype);
    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    ge::Tensor seed;
    op.GetAttr("seed", seed);
    int64_t offset = 0;
    ge::Tensor scalar_offset = CreateScalar(offset, ge::DT_INT64);

    auto data1 = op::Const((ori_name + "data1").c_str()).set_attr_value(seed);
    auto data2 = op::Const((ori_name + "data2").c_str()).set_attr_value(scalar_offset);

    auto statelessbernoulli_v2 = op::StatelessBernoulliV2((ori_name + "StatelessBernoulliV2").c_str())
                                     .set_input_x(data0)
                                     .set_input_seed(data1)
                                     .set_input_offset(data2)
                                     .set_attr_dtype(dtype);

    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(statelessbernoulli_v2, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString(ge::AscendString("ai.onnx::15::Bernoulli")),
                   ge::AscendString(ge::AscendString("ai.onnx::16::Bernoulli")),
                   ge::AscendString(ge::AscendString("ai.onnx::17::Bernoulli")),
                   ge::AscendString(ge::AscendString("ai.onnx::18::Bernoulli"))})
    .ParseParamsByOperatorFn(ParseParamsBernoulli)
    .ParseOpToGraphFn(ParseOpToGraphBernoulli)
    .ImplyType(ImplyType::TVM);
} // namespace domi
