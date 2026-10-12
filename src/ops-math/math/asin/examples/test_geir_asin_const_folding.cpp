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
 * \file test_geir_asin_const_folding.cpp
 * \brief Verifies that Asin is constant folded at graph compile time.
 *
 * Comparing output values alone cannot prove anything here: whether the node is folded on the host or dispatched
 * to the device, the numbers come out the same. So this example asserts on the graph structure instead. After a
 * successful fold GE removes the Asin node and replaces it with a Const carrying the attribute
 * "_is_from_constant_folding". A second run with folding explicitly disabled keeps the assertion honest - if the
 * node disappeared for some unrelated reason, that run fails too.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "gnode.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/asin_proto.h"

namespace {
constexpr int32_t kSuccess = 0;
constexpr int32_t kFailed = -1;
constexpr double kTolerance = 1e-9;
constexpr const char* kOpType = "Asin";
// GE uses "Constant" for the node produced by the folding pass; "Const" is the type of a user built one.
constexpr const char* kConstType = "Const";
constexpr const char* kFoldedConstType = "Constant";
constexpr const char* kFoldedConstAttr = "_is_from_constant_folding";
// GE names a folding-produced Const "dynamic_const_<pid>_<n>" (OpDescUtils::GetConstantOpName).
constexpr const char* kFoldedConstNamePrefix = "dynamic_const_";

// DOUBLE is used because the AICPU implementation of Asin covers FLOAT16/FLOAT/DOUBLE only.
ge::Tensor CreateDoubleTensor(const std::vector<int64_t>& shape, const std::vector<double>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_DOUBLE);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(double));
}

int32_t CreateGraph(ge::Graph& graph)
{
    const std::vector<int64_t> shape = {4};
    const std::vector<double> input_values = {0.0, 0.5, -0.5, 1.0};
    ge::Tensor input_tensor = CreateDoubleTensor(shape, input_values);

    auto input = ge::op::Const("asin_const_input").set_attr_value(input_tensor);
    input.update_output_desc_y(input_tensor.GetTensorDesc());

    auto asin = ge::op::Asin("asin_const_folding").set_input_x(input);
    asin.update_input_desc_x(input_tensor.GetTensorDesc());
    asin.update_output_desc_y(input_tensor.GetTensorDesc());

    graph.SetInputs({input}).SetOutputs({asin});
    return kSuccess;
}

// Counts nodes whose type matches op_type. GetAllNodes walks into subgraphs as well.
int32_t CountNodesOfType(const ge::Graph& graph, const char* op_type)
{
    int32_t count = 0;
    for (const auto& node : graph.GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) != ge::GRAPH_SUCCESS || type.GetString() == nullptr) {
            continue;
        }
        if (std::string(type.GetString()) == op_type) {
            ++count;
        }
    }
    return count;
}

// Detects the Const that replaced the folded node. Two signals are accepted: the "_is_from_constant_folding"
// attribute set by the folding pass, and the generated name prefix used for such Consts. The attribute is not
// always readable through the GNode interface, so the name prefix is the reliable one in practice.
bool HasFoldedConst(const ge::Graph& graph)
{
    for (const auto& node : graph.GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) != ge::GRAPH_SUCCESS || type.GetString() == nullptr) {
            continue;
        }
        const std::string type_str(type.GetString());
        if (type_str != kConstType && type_str != kFoldedConstType) {
            continue;
        }
        bool from_folding = false;
        if (node.GetAttr(ge::AscendString(kFoldedConstAttr), from_folding) == ge::GRAPH_SUCCESS && from_folding) {
            return true;
        }
        ge::AscendString name;
        if (node.GetName(name) == ge::GRAPH_SUCCESS && name.GetString() != nullptr &&
            std::string(name.GetString()).rfind(kFoldedConstNamePrefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

int32_t VerifyOutput(const std::vector<ge::Tensor>& outputs)
{
    const std::vector<double> expected = {0.0, std::asin(0.5), std::asin(-0.5), std::asin(1.0)};
    if (outputs.size() != 1U ||
        outputs[0].GetTensorDesc().GetShape().GetShapeSize() != static_cast<int64_t>(expected.size())) {
        std::printf("Unexpected Asin output shape or count.\n");
        return kFailed;
    }

    const auto* actual = reinterpret_cast<const double*>(outputs[0].GetData());
    if (actual == nullptr) {
        std::printf("Asin output data is null.\n");
        return kFailed;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        std::printf("output[%zu] = %.6f\n", i, actual[i]);
        if (std::fabs(actual[i] - expected[i]) > kTolerance) {
            std::printf("Asin output mismatch at index %zu, expected %.6f.\n", i, expected[i]);
            return kFailed;
        }
    }
    return kSuccess;
}

// Builds, runs and checks one graph. folding_enabled=false disables the pass via ge.oo.constantFolding so that the
// structural assertion can be shown to be non-vacuous.
int32_t RunAndCheck(bool folding_enabled)
{
    ge::Graph graph(folding_enabled ? "asin_constant_folding_test" : "asin_constant_folding_disabled_test");
    if (CreateGraph(graph) != kSuccess) {
        return kFailed;
    }

    std::map<ge::AscendString, ge::AscendString> session_options;
    if (!folding_enabled) {
        session_options[ge::AscendString(ge::OO_CONSTANT_FOLDING)] = ge::AscendString("false");
    }
    ge::Session session(session_options);

    constexpr uint32_t graph_id = 0U;
    const std::map<ge::AscendString, ge::AscendString> graph_options;
    if (session.AddGraph(graph_id, graph, graph_options) != ge::SUCCESS) {
        std::printf("AddGraph failed (folding_enabled=%d).\n", static_cast<int>(folding_enabled));
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graph_id, {}, outputs) != ge::SUCCESS || VerifyOutput(outputs) != kSuccess) {
        std::printf("Asin graph run failed (folding_enabled=%d).\n", static_cast<int>(folding_enabled));
        return kFailed;
    }

    // The folding pass runs during RunGraph, and AddGraph does not copy the graph, so the local object now
    // reflects the optimised graph.
    const int32_t remaining = CountNodesOfType(graph, kOpType);
    const bool folded_const = HasFoldedConst(graph);
    std::printf("[const-folding] folding_enabled=%d, %s nodes left = %d, folded const present = %d\n",
                static_cast<int>(folding_enabled), kOpType, remaining, static_cast<int>(folded_const));

    if (folding_enabled) {
        if (remaining != 0 || !folded_const) {
            std::printf("Asin was NOT constant folded: it is still dispatched to the device. Check that "
                        "math/asin/CMakeLists.txt keeps HOSTCPU TRUE and the kernel uses "
                        "OPS_MATH_REGISTER_CPU_KERNELV2.\n");
            return kFailed;
        }
    } else if (remaining == 0) {
        std::printf("The folding assertion is vacuous: the Asin node disappeared even with constant folding "
                    "disabled, so it proves nothing.\n");
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

    int32_t run_status = RunAndCheck(true);
    if (run_status == kSuccess) {
        run_status = RunAndCheck(false);
    }

    const ge::Status finalize_status = ge::GEFinalize();
    if (run_status != kSuccess || finalize_status != ge::SUCCESS) {
        return kFailed;
    }
    std::printf("Asin constant folding graph passed.\n");
    return kSuccess;
}
