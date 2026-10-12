/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <new>
#include <string>
#include <ctime>
#include <vector>

#include "assert.h"
#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "nn_other.h"
#include "tensor.h"
#include "types.h"

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buffer[64] = {0};
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buffer;
}

uint32_t GetDataTypeSize(ge::DataType type)
{
    switch (type) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4U;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2U;
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8U;
        default:
            return 1U;
    }
}

ge::Tensor MakeFloatTensor(const std::vector<int64_t>& shape)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());

    size_t elem_count = 1U;
    for (auto dim : shape) {
        elem_count *= static_cast<size_t>(dim);
    }
    auto* buffer = new (std::nothrow) float[elem_count];
    for (size_t idx = 0U; idx < elem_count; ++idx) {
        buffer[idx] = static_cast<float>(idx + 1U);
    }
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(buffer), elem_count * sizeof(float));
}

ge::Tensor MakeInt64Tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT64);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());

    auto* buffer = new (std::nothrow) int64_t[values.size()];
    for (size_t idx = 0U; idx < values.size(); ++idx) {
        buffer[idx] = values[idx];
    }
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(buffer), values.size() * sizeof(int64_t));
}

int CreateGraph(
    ge::Graph& graph, std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graph_inputs,
    std::vector<ge::Operator>& graph_outputs)
{
    auto reshape = op::Reshape("reshape");
    auto data = op::Data("x").set_attr_index(0);

    const std::vector<int64_t> x_shape = {2, 3};
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_ND, ge::DT_FLOAT);
    x_desc.SetPlacement(ge::kPlacementHost);
    x_desc.SetFormat(ge::FORMAT_ND);
    x_desc.SetRealDimCnt(x_shape.size());
    data.update_input_desc_x(x_desc);
    data.update_output_desc_y(x_desc);
    inputs.push_back(MakeFloatTensor(x_shape));
    graph.AddOp(data);
    reshape.set_input_x(data);
    graph_inputs.push_back(data);

    auto shape_const = op::Const("shape");
    const std::vector<int64_t> shape_shape = {2};
    ge::TensorDesc shape_desc(ge::Shape(shape_shape), ge::FORMAT_ND, ge::DT_INT64);
    shape_desc.SetPlacement(ge::kPlacementHost);
    shape_desc.SetFormat(ge::FORMAT_ND);
    shape_desc.SetRealDimCnt(shape_shape.size());
    shape_const.SetAttr("value", MakeInt64Tensor(shape_shape, {3, 2}));
    shape_const.update_output_desc_y(shape_desc);
    graph.AddOp(shape_const);
    reshape.set_input_shape(shape_const);
    reshape.update_input_desc_shape(shape_desc);

    ge::TensorDesc y_desc(ge::Shape({3, 2}), ge::FORMAT_ND, ge::DT_FLOAT);
    reshape.update_output_desc_y(y_desc);
    graph_outputs.push_back(reshape);
    return kSuccess;
}
} // namespace

int main()
{
    ge::Graph graph("reshape_geir_test");
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graph_inputs;
    std::vector<ge::Operator> graph_outputs;

    std::cout << GetTime() << " - INFO - initialize ge" << std::endl;
    std::map<ge::AscendString, ge::AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != ge::GRAPH_SUCCESS) {
        std::cerr << GetTime() << " - ERROR - GEInitialize failed" << std::endl;
        return kFailed;
    }

    if (CreateGraph(graph, inputs, graph_inputs, graph_outputs) != kSuccess) {
        std::cerr << GetTime() << " - ERROR - CreateGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);

    std::map<ge::AscendString, ge::AscendString> build_options = {};
    ge::Session session(build_options);
    const uint32_t graph_id = 0U;
    std::map<ge::AscendString, ge::AscendString> graph_options = {};
    if (session.AddGraph(graph_id, graph, graph_options) != ge::GRAPH_SUCCESS) {
        std::cerr << GetTime() << " - ERROR - AddGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graph_id, inputs, outputs) != ge::GRAPH_SUCCESS) {
        std::cerr << GetTime() << " - ERROR - RunGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    if (outputs.empty()) {
        std::cerr << GetTime() << " - ERROR - no output returned" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    const auto& output_desc = outputs[0].GetTensorDesc();
    std::cout << GetTime() << " - INFO - output elements: " << output_desc.GetShape().GetShapeSize() << std::endl;
    std::cout << GetTime() << " - INFO - output bytes: "
              << output_desc.GetShape().GetShapeSize() * GetDataTypeSize(output_desc.GetDataType()) << std::endl;

    ge::GEFinalize();
    return kSuccess;
}