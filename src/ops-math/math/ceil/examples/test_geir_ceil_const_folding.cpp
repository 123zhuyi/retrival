/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/ceil_proto.h"

namespace {
constexpr int32_t kSuccess = 0;
constexpr int32_t kFailed = -1;

ge::Tensor CreateFloatTensor(const std::vector<int64_t>& shape, const std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}

int32_t CreateGraph(ge::Graph& graph)
{
    const std::vector<int64_t> shape = {4};
    const std::vector<float> input_values = {-1.2F, 0.0F, 2.1F, 3.0F};
    ge::Tensor input_tensor = CreateFloatTensor(shape, input_values);

    auto input = ge::op::Const("ceil_const_input").set_attr_value(input_tensor);
    input.update_output_desc_y(input_tensor.GetTensorDesc());

    auto ceil = ge::op::Ceil("ceil_const_folding").set_input_x(input);
    ceil.update_input_desc_x(input_tensor.GetTensorDesc());
    ceil.update_output_desc_y(input_tensor.GetTensorDesc());

    graph.SetInputs({input}).SetOutputs({ceil});
    return kSuccess;
}

int32_t VerifyOutput(const std::vector<ge::Tensor>& outputs)
{
    const std::vector<float> expected = {-1.0F, 0.0F, 3.0F, 3.0F};
    if (outputs.size() != 1U ||
        outputs[0].GetTensorDesc().GetShape().GetShapeSize() != static_cast<int64_t>(expected.size())) {
        std::printf("Unexpected Ceil output shape or count.\n");
        return kFailed;
    }

    const auto* actual = reinterpret_cast<const float*>(outputs[0].GetData());
    if (actual == nullptr) {
        std::printf("Ceil output data is null.\n");
        return kFailed;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        std::printf("output[%zu] = %.1f\n", i, actual[i]);
        if (std::fabs(actual[i] - expected[i]) > 1e-6F) {
            std::printf("Ceil output mismatch at index %zu, expected %.1f.\n", i, expected[i]);
            return kFailed;
        }
    }
    return kSuccess;
}

int32_t RunGraph(const ge::Graph& graph)
{
    const std::map<ge::AscendString, ge::AscendString> session_options;
    ge::Session session(session_options);
    constexpr uint32_t graph_id = 0U;
    const std::map<ge::AscendString, ge::AscendString> graph_options;
    if (session.AddGraph(graph_id, graph, graph_options) != ge::SUCCESS) {
        std::printf("AddGraph failed.\n");
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graph_id, {}, outputs) != ge::SUCCESS || VerifyOutput(outputs) != kSuccess) {
        std::printf("Ceil constant folding graph failed.\n");
        return kFailed;
    }
    return kSuccess;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> global_options = {{"ge.exec.deviceId", "0"},
                                                                         {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != ge::SUCCESS) {
        std::printf("GEInitialize failed.\n");
        return kFailed;
    }

    ge::Graph graph("ceil_constant_folding_test");
    if (CreateGraph(graph) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }

    const int32_t run_status = RunGraph(graph);
    const ge::Status finalize_status = ge::GEFinalize();
    if (run_status != kSuccess || finalize_status != ge::SUCCESS) {
        return kFailed;
    }
    std::printf("Ceil constant folding graph passed.\n");
    return kSuccess;
}
