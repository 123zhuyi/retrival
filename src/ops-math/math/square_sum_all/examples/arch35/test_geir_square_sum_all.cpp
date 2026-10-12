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
 * \file test_geir_square_sum_all.cpp
 * \brief SquareSumAll GE graph example and Ascend 950 precision smoke test.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/square_sum_all_proto.h"

namespace {
constexpr int32_t SUCCESS_CODE = 0;
constexpr int32_t FAILED_CODE = -1;

struct TestCase {
    std::string name;
    std::vector<int64_t> shape;
    float x1Value;
    float x2Value;
};

int64_t GetElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

ge::Tensor MakeHostTensor(const std::vector<int64_t>& shape, std::vector<float>& storage, float value)
{
    storage.assign(static_cast<size_t>(GetElementCount(shape)), value);
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetRealDimCnt(shape.size());
    desc.SetPlacement(ge::kPlacementHost);
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(storage.data()), storage.size() * sizeof(float));
}

bool BuildGraph(const TestCase& testCase, const std::string& graphName, ge::Graph& graph)
{
    const std::string x1Name = graphName + "_x1";
    const std::string x2Name = graphName + "_x2";
    const std::string opName = graphName + "_op";
    auto x1Data = ge::op::Data(x1Name.c_str()).set_attr_index(0);
    auto x2Data = ge::op::Data(x2Name.c_str()).set_attr_index(1);
    ge::TensorDesc inputDesc(ge::Shape(testCase.shape), ge::FORMAT_ND, ge::DT_FLOAT);
    inputDesc.SetPlacement(ge::kPlacementHost);
    x1Data.update_input_desc_x(inputDesc);
    x2Data.update_input_desc_x(inputDesc);
    if (graph.AddOp(x1Data) != ge::GRAPH_SUCCESS || graph.AddOp(x2Data) != ge::GRAPH_SUCCESS) {
        std::printf("[FAIL] %s: failed to add graph inputs\n", testCase.name.c_str());
        return false;
    }

    auto squareSumAll = ge::op::SquareSumAll(opName.c_str());
    squareSumAll.set_input_x1(x1Data);
    squareSumAll.set_input_x2(x2Data);
    const std::vector<int64_t> scalarShape;
    const ge::TensorDesc scalarDesc(ge::Shape(scalarShape), ge::FORMAT_ND, ge::DT_FLOAT);
    squareSumAll.update_output_desc_y1(scalarDesc);
    squareSumAll.update_output_desc_y2(scalarDesc);
    graph.SetInputs(std::vector<ge::Operator>{x1Data, x2Data});
    graph.SetOutputs(std::vector<ge::Operator>{squareSumAll});
    return true;
}

int32_t RunOneCase(ge::Session& session, uint32_t graphId, const TestCase& testCase)
{
    const std::string graphName = "square_sum_all_" + testCase.name;
    ge::Graph graph(graphName.c_str());
    if (!BuildGraph(testCase, graphName, graph)) {
        return FAILED_CODE;
    }

    const std::map<ge::AscendString, ge::AscendString> graphOptions;
    const ge::Status addStatus = session.AddGraph(graphId, graph, graphOptions);
    if (addStatus != ge::SUCCESS) {
        std::printf("[FAIL] %s: AddGraph returned %u\n", testCase.name.c_str(), addStatus);
        return FAILED_CODE;
    }

    std::vector<float> x1Storage;
    std::vector<float> x2Storage;
    std::vector<ge::Tensor> inputs;
    inputs.emplace_back(MakeHostTensor(testCase.shape, x1Storage, testCase.x1Value));
    inputs.emplace_back(MakeHostTensor(testCase.shape, x2Storage, testCase.x2Value));
    std::vector<ge::Tensor> outputs;
    const ge::Status runStatus = session.RunGraph(graphId, inputs, outputs);
    if (runStatus != ge::SUCCESS) {
        std::printf("[FAIL] %s: RunGraph returned %u\n", testCase.name.c_str(), runStatus);
        return FAILED_CODE;
    }
    if (outputs.size() != 2 || outputs[0].GetData() == nullptr || outputs[1].GetData() == nullptr) {
        std::printf("[FAIL] %s: expected two non-null outputs, got %zu\n", testCase.name.c_str(), outputs.size());
        return FAILED_CODE;
    }

    const int64_t elementCount = GetElementCount(testCase.shape);
    const float expectedY1 = static_cast<float>(static_cast<double>(elementCount) * testCase.x1Value *
                                                testCase.x1Value);
    const float expectedY2 = static_cast<float>(static_cast<double>(elementCount) * testCase.x2Value *
                                                testCase.x2Value);
    const float actualY1 = reinterpret_cast<const float*>(outputs[0].GetData())[0];
    const float actualY2 = reinterpret_cast<const float*>(outputs[1].GetData())[0];
    const float toleranceY1 = std::max(1e-5f, std::fabs(expectedY1) * 1e-6f);
    const float toleranceY2 = std::max(1e-5f, std::fabs(expectedY2) * 1e-6f);
    const bool passed = std::fabs(actualY1 - expectedY1) <= toleranceY1 &&
                        std::fabs(actualY2 - expectedY2) <= toleranceY2;
    std::printf("[%s] %-18s N=%-7ld y1=%-12.6f expected=%-12.6f y2=%-12.6f expected=%-12.6f\n",
                passed ? "PASS" : "FAIL", testCase.name.c_str(), elementCount, actualY1, expectedY1, actualY2,
                expectedY2);
    return passed ? SUCCESS_CODE : FAILED_CODE;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::printf("[FAIL] GEInitialize failed\n");
        return FAILED_CODE;
    }

    const std::vector<TestCase> testCases = {
        {"single_element", {1}, 2.0f, -3.0f},     {"tail_63", {63}, 1.0f, -2.0f},
        {"tail_65", {65}, 1.5f, -0.5f},           {"cross_tile_prime", {4099}, 1.0f, 2.0f},
        {"multitile_tail", {327697}, 1.0f, 2.0f}, {"rank_8", {2, 2, 2, 2, 2, 2, 2, 3}, 1.0f, -2.0f},
    };

    int32_t failedCases = 0;
    {
        const std::map<ge::AscendString, ge::AscendString> sessionOptions;
        ge::Session session(sessionOptions);
        for (uint32_t i = 0; i < testCases.size(); ++i) {
            if (RunOneCase(session, i, testCases[i]) != SUCCESS_CODE) {
                ++failedCases;
            }
        }
    }

    if (ge::GEFinalize() != ge::SUCCESS) {
        std::printf("[FAIL] GEFinalize failed\n");
        ++failedCases;
    }
    std::printf("SquareSumAll GE test: %zu cases, %d failures\n", testCases.size(), failedCases);
    return failedCases == 0 ? SUCCESS_CODE : FAILED_CODE;
}
